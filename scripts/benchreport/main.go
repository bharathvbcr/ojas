// benchreport turns the measurement files under bench/results/ into markdown
// summaries and SVG charts. It replaces bench/aggregate.py, bench/plot_all.py,
// bench/plotlib.py and the per-folder summarizers, and reproduces their output
// byte for byte (checked against the Python on every bench/results folder
// before the Python was removed). Run it from the repository root:
//
//	go run scripts/benchreport/main.go aggregate OUT_DIR         # one bench/run_paired.sh run -> markdown on stdout
//	go run scripts/benchreport/main.go aggregate DIR_A DIR_B...  # cross-run direction check
//	go run scripts/benchreport/main.go plots                     # write site/assets/plots/bench/*.svg and docs/bench-plots.md
//	go run scripts/benchreport/main.go plots -check              # exit 1 if any chart or the page is stale
//
// Per-folder summaries (each folder's README names the one it uses):
//
//	go run scripts/benchreport/main.go bench-ops [-parity FILE] LOG...  # ojas-cpu bench_ops.rs / torch_ops.py rounds
//	go run scripts/benchreport/main.go cpu-ab OUT_DIR                   # base/after/torch OJAS_OP, TORCH_OP, OJAS_TAPE rounds
//	go run scripts/benchreport/main.go cpu-ab-linux OUT_DIR             # base/after OJAS_OP rounds per thread count
//	go run scripts/benchreport/main.go attn-ab AB_TXT                   # attention A/B "AB ..." lines
//	go run scripts/benchreport/main.go bpe OUT_DIR BASE_LABEL           # bpe_throughput r<N>_<lane>.txt tables
//	go run scripts/benchreport/main.go decode-ab OUT_DIR [KEY]          # ab_decode.sh before/after JSONL
//	go run scripts/benchreport/main.go sweep-fit OUT_DIR                # sweep_ rows of a run_paired.sh run
//
// aggregate prints the whole summary, then exits 1 when any row is missing,
// crashed or not "ok" in some round, so a runner cannot report a partial run
// as complete. decode-ab does the same for a record that is neither ok nor
// info. Every summary refuses malformed input (exit 1, nothing on stdout).
//
// Every chart shows a time ratio on a log axis where lower is faster, or
// absolute values as dots on a log axis.
//
// Tests: go test scripts/benchreport/main.go scripts/benchreport/main_test.go
//
// It uses only the standard library.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "aggregate":
		os.Exit(runAggregate(os.Args[2:], os.Stdout, os.Stderr))
	case "plots":
		fs := flag.NewFlagSet("plots", flag.ExitOnError)
		check := fs.Bool("check", false, "compare against the files on disk instead of writing them")
		root := fs.String("root", ".", "repository root")
		fs.Parse(os.Args[2:])
		os.Exit(runPlots(*root, *check, os.Stdout, os.Stderr))
	default:
		if s, ok := summaries[os.Args[1]]; ok {
			os.Exit(runSummary(s, os.Args[2:], os.Stdout, os.Stderr))
		}
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: benchreport aggregate OUT_DIR [OUT_DIR...] | plots [-check] [-root DIR] |\n"+
		"       bench-ops [-parity FILE] LOG... | cpu-ab OUT_DIR | cpu-ab-linux OUT_DIR | attn-ab AB_TXT |\n"+
		"       bpe OUT_DIR BASE_LABEL | decode-ab OUT_DIR [KEY] | sweep-fit OUT_DIR")
	os.Exit(2)
}

// ---------------------------------------------------------------- Python-compatible text helpers
//
// The charts and summaries were defined by Python code, and the committed
// files are its output. These helpers keep the exact text: Python's float
// repr, its "g" format, its whitespace set, its universal newlines.

// pyRepr is Python's repr(float): the shortest round-trip digits, fixed
// notation with a ".0" for whole numbers between 1e-4 and 1e16, exponent
// notation outside it.
func pyRepr(f float64) string {
	if math.IsInf(f, 1) {
		return "inf"
	}
	if math.IsInf(f, -1) {
		return "-inf"
	}
	if math.IsNaN(f) {
		return "nan"
	}
	if f == 0 {
		if math.Signbit(f) {
			return "-0.0"
		}
		return "0.0"
	}
	e := strconv.FormatFloat(f, 'e', -1, 64)
	exp, _ := strconv.Atoi(e[strings.IndexByte(e, 'e')+1:])
	if exp < -4 || exp >= 16 {
		return e
	}
	s := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.ContainsRune(s, '.') {
		s += ".0"
	}
	return s
}

// pyG is Python's format(f, ".<prec>g"); prec 6 is a bare "g".
func pyG(f float64, prec int) string { return strconv.FormatFloat(f, 'g', prec, 64) }

func ff(f float64, prec int) string { return strconv.FormatFloat(f, 'f', prec, 64) }

func pyIsSpace(r rune) bool { return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f) }

func pyStrip(s string) string { return strings.TrimFunc(s, pyIsSpace) }

// prefix returns the first n code points of s, as Python's s[:n].
func prefix(s string, n int) string {
	i := 0
	for k := range s {
		if i == n {
			return s[:k]
		}
		i++
	}
	return s
}

// readText reads a file as Python's open() in text mode does: "\r\n" and
// "\r" become "\n".
func readText(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	s := strings.ReplaceAll(string(b), "\r\n", "\n")
	return strings.ReplaceAll(s, "\r", "\n"), nil
}

// fileLines is `for line in open(path)`, with each line's "\n" removed.
func fileLines(path string) ([]string, error) {
	s, err := readText(path)
	if err != nil {
		return nil, err
	}
	if s == "" {
		return nil, nil
	}
	return strings.Split(strings.TrimSuffix(s, "\n"), "\n"), nil
}

// pyFloat parses a string as Python's float() does for the inputs these
// files hold.
func pyFloat(s string) (float64, error) {
	t := strings.ReplaceAll(pyStrip(s), "_", "")
	if strings.ContainsAny(t, "xXpP") && !strings.Contains(strings.ToLower(t), "inf") {
		return 0, fmt.Errorf("could not convert string to float: %q", s)
	}
	f, err := strconv.ParseFloat(t, 64)
	if err != nil && !errors.Is(err, strconv.ErrRange) {
		return 0, fmt.Errorf("could not convert string to float: %q", s)
	}
	return f, nil
}

// globSorted is sorted(glob.glob(pattern)): names starting with "." do not
// match a wildcard.
func globSorted(pattern string) []string {
	m, _ := filepath.Glob(pattern)
	var out []string
	for _, p := range m {
		if !hiddenUnderWildcard(pattern, p) {
			out = append(out, p)
		}
	}
	sort.Strings(out)
	return out
}

func hiddenUnderWildcard(pattern, path string) bool {
	ps := strings.Split(filepath.ToSlash(pattern), "/")
	xs := strings.Split(filepath.ToSlash(path), "/")
	if len(ps) != len(xs) {
		return false
	}
	for i := range ps {
		if strings.HasPrefix(xs[i], ".") && !strings.HasPrefix(ps[i], ".") && strings.ContainsAny(ps[i], "*?[") {
			return true
		}
	}
	return false
}

func median(xs []float64) float64 {
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func minOf(xs []float64) float64 {
	m := xs[0]
	for _, x := range xs[1:] {
		if x < m {
			m = x
		}
	}
	return m
}

func maxOf(xs []float64) float64 {
	m := xs[0]
	for _, x := range xs[1:] {
		if x > m {
			m = x
		}
	}
	return m
}

// log10 matches C's log10 at exact powers of ten, where Go's math.Log10 can
// be one ulp off; a tick-density decision compares against 3.
func log10(x float64) float64 {
	for k := -30; k <= 30; k++ {
		if x == math.Pow10(k) {
			return float64(k)
		}
	}
	return math.Log10(x)
}

// ---------------------------------------------------------------- JSON records
//
// Records keep each number's literal so a value printed as-is reads as
// Python printed it: an integer as "45", a float as "45.0".

type record map[string]any

func decodeRecord(line string) (record, error) {
	dec := json.NewDecoder(strings.NewReader(line))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	if dec.More() {
		return nil, errors.New("extra data after the JSON value")
	}
	m, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("not a JSON object: %s", prefix(line, 80))
	}
	return record(m), nil
}

func isFloatLiteral(n json.Number) bool { return strings.ContainsAny(string(n), ".eE") }

// pyStr is str(value) for a decoded JSON value.
func pyStr(v any) string {
	switch x := v.(type) {
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case string:
		return x
	case json.Number:
		if isFloatLiteral(x) {
			f, _ := strconv.ParseFloat(string(x), 64)
			return pyRepr(f)
		}
		if i, err := strconv.ParseInt(string(x), 10, 64); err == nil {
			return strconv.FormatInt(i, 10)
		}
		return string(x)
	default:
		b, _ := json.Marshal(x)
		return string(b)
	}
}

func (d record) has(k string) bool { _, ok := d[k]; return ok }

func (d record) str(k string) (string, error) {
	v, ok := d[k]
	if !ok {
		return "", fmt.Errorf("KeyError: %q", k)
	}
	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("%q is not a string", k)
	}
	return s, nil
}

func (d record) num(k string) (float64, error) {
	v, ok := d[k]
	if !ok {
		return 0, fmt.Errorf("KeyError: %q", k)
	}
	n, ok := v.(json.Number)
	if !ok {
		return 0, fmt.Errorf("%q is not a number: %v", k, v)
	}
	return strconv.ParseFloat(string(n), 64)
}

// raw is str(d[k]) as Python prints it in an f-string.
func (d record) raw(k string) (string, error) {
	v, ok := d[k]
	if !ok {
		return "", fmt.Errorf("KeyError: %q", k)
	}
	return pyStr(v), nil
}

// get is d.get(k, "") rendered as text.
func (d record) get(k string) string {
	v, ok := d[k]
	if !ok {
		return ""
	}
	return pyStr(v)
}

// ---------------------------------------------------------------- aggregate: one bench/run_paired.sh run
//
// Per row and ojas backend, each round gives one ojas median and one torch
// median. ratio = torch_median / ojas_median for that round, so > 1 means ojas
// was faster. Reported: ojas and torch min (min over rounds of the per-round
// minimum) and median (median of the per-round medians), the median of the
// per-round ratios with its min and max, and the within-run spread of each
// side (max / min of its per-round medians). A row whose spread exceeds 10% on
// either side is flagged "noisy - not quoted". A row that is missing or not
// "ok" on either side in any round is reported with its status, never dropped.

const spreadLimit = 0.10

var ojasRuntimes = []string{"ojas-metal", "ojas-wgpu", "ojas-cpu", "ojas-cpu-host"}

// decodeOnly lanes run only the decode rows (bench/decode_rows.rs).
var decodeOnly = map[string]bool{"ojas-cpu": true, "ojas-cpu-host": true}

const genPrefix = "gen_"
const torchRT = "torch-mps"

type rkey struct{ rt, row string }

type round struct {
	keys []rkey // first-insertion order
	recs map[rkey]record
}

type runData struct {
	order []int // first-seen order of rounds
	by    map[int]*round
}

func (rs *runData) sorted() []int {
	s := append([]int(nil), rs.order...)
	sort.Ints(s)
	return s
}

func (rs *runData) get(r int, k rkey) (record, bool) {
	d, ok := rs.by[r].recs[k]
	return d, ok
}

func roundOf(path string) (int, error) {
	b := filepath.Base(filepath.Dir(path))
	n, err := strconv.Atoi(strings.TrimFunc(b[min(5, len(b)):], pyIsSpace))
	if err != nil {
		return 0, fmt.Errorf("round directory %q: %v", b, err)
	}
	return n, nil
}

func loadRounds(out string) (*runData, error) {
	rs := &runData{by: map[int]*round{}}
	for _, path := range globSorted(filepath.Join(out, "round*", "*.jsonl")) {
		r, err := roundOf(path)
		if err != nil {
			return nil, err
		}
		lines, err := fileLines(path)
		if err != nil {
			return nil, err
		}
		for i, line := range lines {
			line = pyStrip(line)
			if line == "" {
				continue
			}
			d, err := decodeRecord(line)
			if err != nil {
				return nil, fmt.Errorf("%s:%d: %v", path, i+1, err)
			}
			rt, err := d.str("runtime")
			if err != nil {
				return nil, fmt.Errorf("%s:%d: %v", path, i+1, err)
			}
			row, err := d.str("row")
			if err != nil {
				return nil, fmt.Errorf("%s:%d: %v", path, i+1, err)
			}
			// An info record about a row (`_gen_traffic` for each decode row)
			// is keyed by that row too, so one per row survives.
			if d.has("for") {
				row = row + "@" + d.get("for")
			}
			rd := rs.by[r]
			if rd == nil {
				rd = &round{recs: map[rkey]record{}}
				rs.by[r] = rd
				rs.order = append(rs.order, r)
			}
			k := rkey{rt, row}
			if _, ok := rd.recs[k]; !ok {
				rd.keys = append(rd.keys, k)
			}
			rd.recs[k] = d
		}
	}
	return rs, nil
}

func fmtMS(x float64) string {
	switch {
	case x >= 100:
		return ff(x, 1)
	case x >= 10:
		return ff(x, 2)
	}
	return ff(x, 3)
}

func rowNames(rs *runData, rt string) []string {
	var names []string
	seen := map[string]bool{}
	for _, r := range rs.order {
		for _, k := range rs.by[r].keys {
			if k.rt == rt && !strings.HasPrefix(k.row, "_") && !seen[k.row] {
				seen[k.row] = true
				names = append(names, k.row)
			}
		}
	}
	return names
}

type summary struct {
	row, torchRow                       string
	problems                            []string
	parity, rel                         *float64
	n                                   int
	oMin, oMed, tMin, tMed              float64
	ratio, rmin, rmax, oSpread, tSpread float64
	noisy                               bool
}

func summarize(rs *runData, rt, row, torchRow string) (*summary, error) {
	var oMeds, tMeds, oMins, tMins, ratios, parity, rel []float64
	var problems []string
	for _, r := range rs.sorted() {
		o, okO := rs.get(r, rkey{rt, row})
		t, okT := rs.get(r, rkey{torchRT, torchRow})
		if !okO {
			if crash, ok := rs.get(r, rkey{rt, "_process"}); ok {
				ex, err := crash.raw("exit")
				if err != nil {
					return nil, err
				}
				problems = append(problems, fmt.Sprintf("r%d ojas: crash exit %s", r, ex))
			} else {
				problems = append(problems, fmt.Sprintf("r%d ojas: NO DATA", r))
			}
			continue
		}
		if v, ok := o["parity_max_abs"]; ok && v != nil {
			p, err := o.num("parity_max_abs")
			if err != nil {
				return nil, err
			}
			q, err := o.num("parity_rel")
			if err != nil {
				return nil, err
			}
			parity = append(parity, p)
			rel = append(rel, q)
		}
		st, err := o.raw("status")
		if err != nil {
			return nil, err
		}
		if st != "ok" {
			problems = append(problems, prefix(fmt.Sprintf("r%d ojas %s: %s", r, st, o.get("detail")), 300))
			continue
		}
		tst := ""
		if okT {
			if tst, err = t.str("status"); err != nil {
				return nil, err
			}
		}
		if !okT || tst != "ok" {
			what := "NO DATA"
			if okT {
				what = tst + " " + t.get("detail")
			}
			problems = append(problems, prefix(fmt.Sprintf("r%d torch: %s", r, what), 300))
			continue
		}
		var om, tm, on, tn float64
		for _, f := range []struct {
			d   record
			k   string
			dst *float64
		}{{o, "median_ms", &om}, {t, "median_ms", &tm}, {o, "min_ms", &on}, {t, "min_ms", &tn}} {
			if *f.dst, err = f.d.num(f.k); err != nil {
				return nil, err
			}
		}
		oMeds, tMeds, oMins, tMins = append(oMeds, om), append(tMeds, tm), append(oMins, on), append(tMins, tn)
		ratios = append(ratios, tm/om)
	}
	s := &summary{row: row, torchRow: torchRow, problems: problems, n: len(ratios)}
	if len(parity) > 0 {
		p, q := maxOf(parity), maxOf(rel)
		s.parity, s.rel = &p, &q
	}
	if len(ratios) > 0 {
		s.oMin, s.oMed = minOf(oMins), median(oMeds)
		s.tMin, s.tMed = minOf(tMins), median(tMeds)
		s.ratio, s.rmin, s.rmax = median(ratios), minOf(ratios), maxOf(ratios)
		s.oSpread = maxOf(oMeds)/minOf(oMeds) - 1
		s.tSpread = maxOf(tMeds)/minOf(tMeds) - 1
		s.noisy = s.oSpread > spreadLimit || s.tSpread > spreadLimit
	}
	return s, nil
}

func (s *summary) name() string {
	if s.torchRow == s.row {
		return s.row
	}
	return s.row + " vs " + s.torchRow
}

func table(rows []*summary, rt string) []string {
	out := []string{"### " + rt + " vs " + torchRT, "",
		"| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |",
		"| :-- | --: | --: | :-- | :-- | :-- | :-- |"}
	for _, s := range rows {
		par := "-"
		if s.parity != nil {
			par = fmt.Sprintf("%.2e (%.1e)", *s.parity, *s.rel)
		}
		if s.n == 0 {
			out = append(out, fmt.Sprintf("| %s | - | - | - | - | **%s** | %s |", s.name(), strings.Join(s.problems, "; "), par))
			continue
		}
		flag := ""
		if s.noisy {
			flag = "noisy - not quoted"
		}
		if len(s.problems) > 0 {
			if flag != "" {
				flag += "; "
			}
			flag += fmt.Sprintf("partial (%d rounds): ", s.n) + strings.Join(s.problems, "; ")
		}
		out = append(out, fmt.Sprintf("| %s | %s / %s | %s / %s | %.2fx [%.2f-%.2f] | %s%% / %s%% | %s | %s |",
			s.name(), fmtMS(s.oMin), fmtMS(s.oMed), fmtMS(s.tMin), fmtMS(s.tMed),
			s.ratio, s.rmin, s.rmax, ff(100*s.oSpread, 0), ff(100*s.tSpread, 0), flag, par))
	}
	return out
}

// aggregateRun renders one run directory. incomplete counts the rows that
// have a problem in some round or no data at all.
func aggregateRun(out string) (text string, incomplete int, err error) {
	rs, err := loadRounds(out)
	if err != nil {
		return "", 0, err
	}
	lines := []string{"# Paired GPU-vs-torch summary: " + out, "",
		fmt.Sprintf("%d rounds. ratio = torch median / ojas median per round ", len(rs.order)) +
			"(> 1: ojas faster). Spread = max/min of that side's per-round medians.", ""}
	env := filepath.Join(out, "env.txt")
	if _, serr := os.Stat(env); serr == nil {
		t, err := readText(env)
		if err != nil {
			return "", 0, err
		}
		lines = append(lines, "```", strings.TrimRightFunc(t, pyIsSpace), "```", "")
	}
	torchRows := rowNames(rs, torchRT)
	torchHas := map[string]bool{}
	for _, n := range torchRows {
		torchHas[n] = true
	}
	present := map[string]bool{}
	for _, r := range rs.order {
		for _, k := range rs.by[r].keys {
			present[k.rt] = true
		}
	}
	type ranked struct {
		ratio float64
		rt    string
		s     *summary
	}
	var rank []ranked
	for _, rt := range ojasRuntimes {
		if !present[rt] {
			continue
		}
		var rows []*summary
		names := rowNames(rs, rt)
		have := map[string]bool{}
		for _, n := range names {
			have[n] = true
			s, err := summarize(rs, rt, n, n)
			if err != nil {
				return "", 0, err
			}
			rows = append(rows, s)
			if torchHas[n+"_bf16"] {
				s, err := summarize(rs, rt, n, n+"_bf16")
				if err != nil {
					return "", 0, err
				}
				rows = append(rows, s)
			}
		}
		for _, n := range torchRows {
			if decodeOnly[rt] && !strings.HasPrefix(n, genPrefix) {
				continue
			}
			if !have[n] && !strings.HasSuffix(n, "_bf16") {
				rows = append(rows, &summary{row: n, torchRow: n, problems: []string{"NO DATA: row absent from " + rt}})
			}
		}
		lines = append(append(lines, table(rows, rt)...), "")
		for _, s := range rows {
			if s.n == 0 || len(s.problems) > 0 {
				incomplete++
			}
			if s.n > 0 && !strings.HasPrefix(s.row, "floor") {
				rank = append(rank, ranked{s.ratio, rt, s})
			}
		}
	}
	sort.SliceStable(rank, func(i, j int) bool { return rank[i].ratio < rank[j].ratio })
	lines = append(lines, "### Ranked: where ojas is slowest relative to torch (lowest ratio first)", "",
		"| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |",
		"| --: | :-- | :-- | :-- | --: | --: | :-- |")
	for i, x := range rank {
		flag := ""
		if x.s.noisy {
			flag = "noisy"
		}
		lines = append(lines, fmt.Sprintf("| %d | %s | %s | %.2fx [%.2f-%.2f] | %s | %s | %s |",
			i+1, x.rt, x.s.name(), x.ratio, x.s.rmin, x.s.rmax, fmtMS(x.s.oMed), fmtMS(x.s.tMed), flag))
	}
	load := filepath.Join(out, "load.jsonl")
	if _, serr := os.Stat(load); serr == nil {
		lines = append(lines, "", "### Machine load around each runtime block", "",
			"| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |",
			"| --: | :-- | :-- | :-- | :-- | --: |")
		ll, err := fileLines(load)
		if err != nil {
			return "", 0, err
		}
		for i, line := range ll {
			d, err := decodeRecord(line)
			if err != nil {
				return "", 0, fmt.Errorf("%s:%d: %v", load, i+1, err)
			}
			var cells []string
			for _, k := range []string{"round", "lane", "when", "time", "load", "gpu_util_pct"} {
				c, err := d.raw(k)
				if err != nil {
					return "", 0, fmt.Errorf("%s:%d: %v", load, i+1, err)
				}
				cells = append(cells, c)
			}
			lines = append(lines, "| "+strings.Join(cells, " | ")+" |")
		}
	}
	dr, err := decodeRates(rs)
	if err != nil {
		return "", 0, err
	}
	pl, err := perRowLoad(out)
	if err != nil {
		return "", 0, err
	}
	lines = append(append(lines, dr...), pl...)
	if len(rs.order) == 0 {
		incomplete++
	}
	return strings.Join(lines, "\n") + "\n", incomplete, nil
}

var decodeRow = regexp.MustCompile(`^gen_((?:[a-z0-9]+_)?)((?:greedy|topk\d+|temp0))_p(\d+)_n(\d+)$`)

// decodeRates: decode tokens/second per runtime and decode row. Per-token ms
// = (decode median - prefill median) / (N - 1), where a row
// `gen_[<model>_]<kind>_p<P>_n<N>` (kind `greedy`, `topk<K>` or `temp0`)
// pairs with `gen_[<model>_]prefill_p<P>`; the median over rounds. Plus the
// last round's `_gen_traffic` and `_gen_ids` records for the row, and each
// runtime's `_gen_readback` record.
func decodeRates(rs *runData) ([]string, error) {
	var out, readback []string
	set := map[string]bool{}
	for _, r := range rs.order {
		for _, k := range rs.by[r].keys {
			if decodeRow.MatchString(k.row) {
				set[k.row] = true
			}
		}
	}
	var names []string
	for n := range set {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, rt := range append([]string{torchRT}, ojasRuntimes...) {
		for _, name := range names {
			g := decodeRow.FindStringSubmatch(name)
			model, p, nw := g[1], g[3], g[4]
			newTok, _ := strconv.Atoi(nw)
			prefillName := "gen_" + model + "prefill_p" + p
			var perTok, prefill []float64
			extra := map[string]record{}
			for _, r := range rs.sorted() {
				for _, kind := range []string{"_gen_traffic", "_gen_ids"} {
					if d, ok := rs.get(r, rkey{rt, kind + "@" + name}); ok {
						extra[kind] = d
					}
				}
				dec, okD := rs.get(r, rkey{rt, name})
				pre, okP := rs.get(r, rkey{rt, prefillName})
				if !okD || !okP || len(dec) == 0 || len(pre) == 0 || dec.get("status") != "ok" || pre.get("status") != "ok" {
					continue
				}
				dm, err := dec.num("median_ms")
				if err != nil {
					return nil, err
				}
				pm, err := pre.num("median_ms")
				if err != nil {
					return nil, err
				}
				if newTok == 1 {
					return nil, fmt.Errorf("ZeroDivisionError: decode row %s generates one token", name)
				}
				perTok = append(perTok, (dm-pm)/float64(newTok-1))
				prefill = append(prefill, pm)
			}
			if len(perTok) == 0 {
				continue
			}
			ms := median(perTok)
			rate := "-"
			if ms > 0 {
				rate = ff(1e3/ms, 1)
			}
			cells := []string{rt, name, fmtMS(median(prefill)), fmtMS(ms), rate, strconv.Itoa(len(perTok))}
			t := extra["_gen_traffic"]
			if !t.has("step_uploads") {
				cells = append(cells, "-")
			} else {
				var v [3]float64
				for i, k := range []string{"step_uploads", "step_upload_bytes", "step_readback_bytes"} {
					x, err := t.num(k)
					if err != nil {
						return nil, err
					}
					v[i] = x
				}
				cells = append(cells, fmt.Sprintf("%s / %s B / %s B", ff(v[0], 0), ff(v[1], 0), ff(v[2], 0)))
			}
			ids := extra["_gen_ids"]
			if !ids.has("ids_equal") {
				cells = append(cells, "-")
			} else {
				var v [3]string
				for i, k := range []string{"ids_equal", "of", "common_prefix"} {
					x, err := ids.raw(k)
					if err != nil {
						return nil, err
					}
					v[i] = x
				}
				cells = append(cells, fmt.Sprintf("%s/%s (prefix %s)", v[0], v[1], v[2]))
			}
			out = append(out, "| "+strings.Join(cells, " | ")+" |")
		}
		if len(rs.order) > 0 {
			last := rs.sorted()[len(rs.order)-1]
			keys := append([]rkey(nil), rs.by[last].keys...)
			sort.Slice(keys, func(i, j int) bool {
				if keys[i].rt != keys[j].rt {
					return keys[i].rt < keys[j].rt
				}
				return keys[i].row < keys[j].row
			})
			for _, k := range keys {
				d := rs.by[last].recs[k]
				if k.rt != rt || !strings.HasPrefix(k.row, "_gen_readback@") || !d.has("to_host_us") {
					continue
				}
				var f [3]float64
				for i, n := range []string{"to_host_us", "read_bytes_us", "decode_pass_us"} {
					x, err := d.num(n)
					if err != nil {
						return nil, err
					}
					f[i] = x
				}
				by, err := d.raw("bytes")
				if err != nil {
					return nil, err
				}
				it, err := d.raw("iters")
				if err != nil {
					return nil, err
				}
				readback = append(readback, fmt.Sprintf("| %s | %s | %.2f | %.2f | %.2f | %s |", rt, by, f[0], f[1], f[2], it))
			}
		}
	}
	if len(out) == 0 {
		return nil, nil
	}
	lines := append([]string{"", "### Decode (gen_ rows)", "",
		"Per-token ms = (decode median - prefill median) / (N - 1), median over rounds, " +
			"each decode row against its model's prefill row. Traffic is per decode step, from " +
			"DeviceDecoder's counters (uploads / bytes up / bytes read back).", "",
		"| runtime | row | prefill median ms | decode ms / token | tokens / s | rounds " +
			"| step traffic | greedy ids equal to torch |",
		"| :-- | :-- | --: | --: | --: | --: | :-- | :-- |"}, out...)
	if len(readback) > 0 {
		lines = append(append(lines, "", "Reading one `[1, vocab]` logit row back (last round, min of `iters`): "+
			"`Tensor::to_host` against `DeviceBuffer::read_bytes` alone; the difference is "+
			"the transient decode pass a read into a caller's buffer would save.", "",
			"| runtime | bytes | to_host us | read_bytes us | decode pass us | iters |",
			"| :-- | --: | --: | --: | --: | --: |"), readback...)
	}
	return lines, nil
}

// gpuSample keeps a GPU utilisation value with the text Python printed for it.
type gpuSample struct {
	f float64
	s string
}

// perRowLoad: range of the per-row `_load` records (uptime 1-min load and
// the ioreg GPU sample, before and after every row) per round and lane.
func perRowLoad(out string) ([]string, error) {
	var rows []string
	for _, path := range globSorted(filepath.Join(out, "round*", "*.jsonl")) {
		r, err := roundOf(path)
		if err != nil {
			return nil, err
		}
		base := filepath.Base(path)
		lane := base[:max(len(base)-6, 0)]
		var loads []float64
		var gpus []gpuSample
		ll, err := fileLines(path)
		if err != nil {
			return nil, err
		}
		for i, line := range ll {
			line = pyStrip(line)
			if line == "" {
				continue
			}
			d, err := decodeRecord(line)
			if err != nil {
				return nil, fmt.Errorf("%s:%d: %v", path, i+1, err)
			}
			if v, ok := d["row"]; !ok || v != "_load" {
				continue
			}
			l, err := d.str("load")
			if err != nil {
				return nil, fmt.Errorf("%s:%d: %v", path, i+1, err)
			}
			if fs := strings.FieldsFunc(strings.ReplaceAll(l, ",", " "), pyIsSpace); len(fs) > 0 {
				if f, err := pyFloat(fs[0]); err == nil {
					loads = append(loads, f)
				}
			}
			if v, ok := d["gpu_util_pct"]; ok && v != nil {
				f, err := d.num("gpu_util_pct")
				if err != nil {
					return nil, fmt.Errorf("%s:%d: %v", path, i+1, err)
				}
				gpus = append(gpus, gpuSample{f, pyStr(v)})
			}
		}
		if len(loads) == 0 && len(gpus) == 0 {
			continue
		}
		lr, gr := "-", "-"
		if len(loads) > 0 {
			lr = ff(minOf(loads), 1) + "-" + ff(maxOf(loads), 1)
		}
		if len(gpus) > 0 {
			lo, hi := gpus[0], gpus[0]
			fs := make([]float64, len(gpus))
			for i, g := range gpus {
				fs[i] = g.f
				if g.f < lo.f {
					lo = g
				}
				if g.f > hi.f {
					hi = g
				}
			}
			gr = lo.s + " / " + ff(median(fs), 0) + " / " + hi.s
		}
		rows = append(rows, fmt.Sprintf("| %d | %s | %d | %s | %s |", r, lane, len(loads), lr, gr))
	}
	if len(rows) == 0 {
		return nil, nil
	}
	return append([]string{"", "### Load recorded before and after every row", "",
		"| round | lane | samples | 1-min load min-max | GPU util % min / median / max |",
		"| --: | :-- | --: | :-- | :-- |"}, rows...), nil
}

// crossRun: several run directories. Per row, each run's median ratio
// [min-max] and a verdict over every round of every run. "slower in all N" /
// "faster in all N" holds only when every per-round ratio is on one side of 1.
func crossRun(outs []string) (string, int, error) {
	var runs []*runData
	for _, o := range outs {
		rs, err := loadRounds(o)
		if err != nil {
			return "", 0, err
		}
		runs = append(runs, rs)
	}
	torchHas := map[string]bool{}
	for _, n := range rowNames(runs[0], torchRT) {
		torchHas[n] = true
	}
	labels := make([]string, len(outs))
	for i := range outs {
		labels[i] = string(rune('A' + i))
	}
	incomplete := 0
	lines := []string{"# Cross-run direction check", ""}
	for i, o := range outs {
		lines = append(lines, "- run "+labels[i]+": "+o)
	}
	lines = append(lines, "")
	for _, rt := range ojasRuntimes {
		hdr, sep := "| row | ", "| :-- | "
		var hs, ss []string
		for _, l := range labels {
			hs = append(hs, "run "+l+" median [min-max]")
			ss = append(ss, ":--")
		}
		lines = append(lines, "### "+rt+" vs "+torchRT, "",
			hdr+strings.Join(hs, " | ")+" | all rounds min-max | verdict |",
			sep+strings.Join(ss, " | ")+" | :-- | :-- |")
		for _, n := range rowNames(runs[0], rt) {
			trs := []string{n}
			if torchHas[n+"_bf16"] {
				trs = append(trs, n+"_bf16")
			}
			for _, tr := range trs {
				var cells []string
				var all []float64
				missing := 0
				for _, rd := range runs {
					s, err := summarize(rd, rt, n, tr)
					if err != nil {
						return "", 0, err
					}
					if s.n == 0 {
						cells = append(cells, "NO DATA")
						missing++
						continue
					}
					cells = append(cells, fmt.Sprintf("%.2fx [%.2f-%.2f]", s.ratio, s.rmin, s.rmax))
					for _, r := range rd.sorted() {
						o, okO := rd.get(r, rkey{rt, n})
						t, okT := rd.get(r, rkey{torchRT, tr})
						if okO && okT && len(o) > 0 && len(t) > 0 && o.get("status") == "ok" && t.get("status") == "ok" {
							om, err := o.num("median_ms")
							if err != nil {
								return "", 0, err
							}
							tm, err := t.num("median_ms")
							if err != nil {
								return "", 0, err
							}
							all = append(all, tm/om)
						}
					}
				}
				verdict := "mixed"
				if len(all) > 0 && maxOf(all) < 1 {
					verdict = fmt.Sprintf("ojas slower in all %d rounds", len(all))
				} else if len(all) > 0 && minOf(all) > 1 {
					verdict = fmt.Sprintf("ojas faster in all %d rounds", len(all))
				}
				if missing > 0 {
					verdict += fmt.Sprintf(" (%d run(s) without data)", missing)
					incomplete++
				}
				span := "-"
				if len(all) > 0 {
					span = fmt.Sprintf("%.2f-%.2f", minOf(all), maxOf(all))
				}
				name := n
				if tr != n {
					name = n + " vs " + tr
				}
				lines = append(lines, "| "+name+" | "+strings.Join(cells, " | ")+" | "+span+" | "+verdict+" |")
			}
		}
		lines = append(lines, "")
	}
	return strings.Join(lines, "\n") + "\n", incomplete, nil
}

func runAggregate(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: benchreport aggregate OUT_DIR [OUT_DIR...]")
		return 2
	}
	var text string
	var incomplete int
	var err error
	if len(args) > 1 {
		text, incomplete, err = crossRun(args)
	} else {
		text, incomplete, err = aggregateRun(args[0])
	}
	if err != nil {
		fmt.Fprintln(stderr, "aggregate:", err)
		return 1
	}
	io.WriteString(stdout, text)
	if incomplete > 0 {
		fmt.Fprintf(stderr, "aggregate: %d row(s) incomplete (missing, crashed or not ok in some round); see the flags in the summary\n", incomplete)
		return 1
	}
	return 0
}

// ---------------------------------------------------------------- SVG charts

const (
	green, grey, red, blue, orange = "#2a9d8f", "#9aa0a6", "#d1495b", "#3d6fb6", "#e08e2b"
	font                           = `font-family="Helvetica,Arial,sans-serif" font-size="12"`
)

var palette = []string{blue, orange, green, red, "#7b5ea7", "#8c564b", "#17becf", "#bcbd22"}

func esc(s string) string {
	return strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;").Replace(s)
}

func verdictColor(v, band float64) string {
	if v < 1/band {
		return green
	}
	if v > band {
		return red
	}
	return grey
}

func ticks(lo, hi float64) []float64 {
	mults := []float64{1, 2, 5}
	if !(log10(hi/lo) <= 3) {
		mults = []float64{1}
	}
	var out []float64
	for e := int(math.Floor(log10(lo))); math.Pow10(e) <= hi*1.001; e++ {
		for _, m := range mults {
			t := m * math.Pow10(e)
			if lo*0.999 <= t && t <= hi*1.001 {
				out = append(out, t)
			}
		}
	}
	return out
}

// ratioRow is one bar: v is a time ratio (< 1 faster); lo/hi a whisker, mid
// a median marker, hollow a row the run marked noisy.
type ratioRow struct {
	name            string
	v               float64
	lo, hi, mid     *float64
	hollow          bool
	color           string
	ojasMS, torchMS float64
}

func fp(f float64) *float64 { return &f }

// The float64 conversions around products keep Go from fusing a*b+c into one
// FMA on arm64; Python rounds the product first, and so must this.

func ratioChart(rows []ratioRow, title, subtitle string, band float64) (string, bool) {
	if len(rows) == 0 {
		return "", false
	}
	var vals []float64
	for _, r := range rows {
		vals = append(vals, r.v)
	}
	for _, r := range rows {
		if r.lo != nil {
			vals = append(vals, *r.lo)
		} else {
			vals = append(vals, r.v)
		}
	}
	for _, r := range rows {
		if r.hi != nil {
			vals = append(vals, *r.hi)
		} else {
			vals = append(vals, r.v)
		}
	}
	var pos []float64
	for _, v := range vals {
		if v != 0 && v > 0 {
			pos = append(pos, v)
		}
	}
	lo := math.Min(minOf(pos)*0.55, 0.8)
	hi := math.Max(maxOf(pos)*1.25, 1.25)
	lo, hi = math.Max(lo, 1e-3), math.Min(hi, 1e3)
	const labelW, plotW = 300, 560
	rh, top, right := 20, 78, 60
	W, H := labelW+plotW+right, top+rh*len(rows)+50
	llo, lhi := math.Log(lo), math.Log(hi)
	x := func(v float64) float64 {
		c := v
		if lo > c {
			c = lo
		}
		if hi < c {
			c = hi
		}
		return float64(labelW) + float64((math.Log(c)-llo)/(lhi-llo)*float64(plotW))
	}
	s := []string{fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" %s>`, W, H, W, H, font),
		`<rect width="100%" height="100%" fill="#fff"/>`,
		`<text x="16" y="24" font-size="15" font-weight="bold" fill="#111">` + esc(title) + `</text>`,
		`<text x="16" y="44" fill="#555">` + esc(subtitle) + `</text>`,
		fmt.Sprintf(`<text x="16" y="62" fill="#555">Lower is faster. Grey band = within %d%%. `, int((band-1)*100)) +
			`Hollow bar = run marked noisy. Whisker = per-round min..max.</text>`}
	bot := top + rh*len(rows)
	s = append(s, fmt.Sprintf(`<rect x="%s" y="%d" width="%s" height="%d" fill="#eee"/>`, ff(x(1/band), 1), top-6, ff(x(band)-x(1/band), 1), bot-top+6))
	for _, t := range ticks(lo, hi) {
		stroke, sw := "#ddd", "1"
		if math.Abs(t-1) < 1e-9 {
			stroke, sw = "#222", "1.5"
		}
		s = append(s, fmt.Sprintf(`<line x1="%s" x2="%s" y1="%d" y2="%d" stroke="%s" stroke-width="%s"/>`, ff(x(t), 1), ff(x(t), 1), top-6, bot, stroke, sw))
		s = append(s, fmt.Sprintf(`<text x="%s" y="%d" text-anchor="middle" fill="#444">%sx</text>`, ff(x(t), 1), bot+16, pyG(t, 6)))
	}
	for i, r := range rows {
		y := top + i*rh
		v := r.v
		c := r.color
		if c == "" {
			c = verdictColor(v, band)
		}
		x0, x1 := x(1), x(v)
		if x1 < x0 {
			x0, x1 = x1, x0
		}
		s = append(s, fmt.Sprintf(`<text x="%d" y="%d" text-anchor="end" fill="#222">%s</text>`, labelW-8, y+14, esc(prefix(r.name, 46))))
		width := ff(math.Max(x1-x0, 1), 1)
		if x1-x0 < 1 {
			width = "1.0"
		}
		if r.hollow {
			s = append(s, fmt.Sprintf(`<rect x="%s" y="%d" width="%s" height="%d" fill="#fff" stroke="%s" stroke-width="1.4"/>`, ff(x0, 1), y+3, width, rh-7, c))
		} else {
			s = append(s, fmt.Sprintf(`<rect x="%s" y="%d" width="%s" height="%d" fill="%s"/>`, ff(x0, 1), y+3, width, rh-7, c))
		}
		if r.lo != nil && r.hi != nil {
			yc := float64(y) + float64(rh)/2 - 0.5
			s = append(s, fmt.Sprintf(`<line x1="%s" x2="%s" y1="%s" y2="%s" stroke="#222" stroke-width="1"/>`, ff(x(*r.lo), 1), ff(x(*r.hi), 1), ff(yc, 1), ff(yc, 1)))
			for _, e := range []float64{*r.lo, *r.hi} {
				s = append(s, fmt.Sprintf(`<line x1="%s" x2="%s" y1="%s" y2="%s" stroke="#222" stroke-width="1"/>`, ff(x(e), 1), ff(x(e), 1), ff(yc-3, 1), ff(yc+3, 1)))
			}
		}
		if r.mid != nil {
			s = append(s, fmt.Sprintf(`<circle cx="%s" cy="%s" r="3.2" fill="#fff" stroke="#222" stroke-width="1.2"/>`, ff(x(*r.mid), 1), ff(float64(y)+float64(rh)/2-0.5, 1)))
		}
		var tx float64
		anchor := "end"
		if v >= 1 {
			hv := v
			if r.hi != nil {
				hv = *r.hi
			}
			tx = math.Max(x(hv), x1) + 5
			anchor = "start"
		} else {
			lv := v
			if r.lo != nil {
				lv = *r.lo
			}
			tx = math.Min(x(lv), x0) - 5
		}
		s = append(s, fmt.Sprintf(`<text x="%s" y="%d" text-anchor="%s" fill="#333">%s</text>`, ff(tx, 1), y+14, anchor, ff(v, 2)))
	}
	s = append(s, "</svg>")
	return strings.Join(s, "\n"), true
}

type group struct {
	label string
	vals  []*float64
}

type series struct{ name, color string }

// pyNum is a Python number that may still be an int, so that it prints as
// Python printed it ("22" against "103.6").
type pyNum struct {
	f     float64
	isInt bool
}

func (n pyNum) add(v float64) pyNum { return pyNum{n.f + v, false} }

func (n pyNum) String() string {
	if n.isInt {
		return strconv.FormatInt(int64(n.f), 10)
	}
	return pyRepr(n.f)
}

// groupedChart draws grouped dots on a log axis in absolute units.
func groupedChart(groups []group, ser []series, title, subtitle, unit string) (string, bool) {
	var vals []float64
	for _, g := range groups {
		for _, v := range g.vals {
			if v != nil && *v != 0 {
				vals = append(vals, *v)
			}
		}
	}
	if len(vals) == 0 {
		return "", false
	}
	lo, hi := minOf(vals)*0.7, maxOf(vals)*1.6
	const labelW, plotW = 300, 520
	n := len(ser)
	bh, gap, right := 11, 9, 70
	gh := bh*n + gap
	W := labelW + plotW + right
	// legend: items wrap onto new rows instead of running off the canvas
	type lp struct {
		x pyNum
		y int
	}
	var pos []lp
	lx, ly := pyNum{16, true}, 0
	for _, se := range ser {
		w := float64(24) + float64(float64(utf8.RuneCountInString(se.name))*6.4) + 18
		if lx.f+w > float64(W-16) && lx.f > 16 {
			lx, ly = pyNum{16, true}, ly+18
		}
		pos = append(pos, lp{lx, ly})
		lx = lx.add(w)
	}
	top := 62 + ly + 22
	H := top + gh*len(groups) + 50
	llo, lhi := math.Log(lo), math.Log(hi)
	x := func(v float64) float64 {
		return float64(labelW) + float64((math.Log(v)-llo)/(lhi-llo)*float64(plotW))
	}
	s := []string{fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" %s>`, W, H, W, H, font),
		`<rect width="100%" height="100%" fill="#fff"/>`,
		`<text x="16" y="24" font-size="15" font-weight="bold" fill="#111">` + esc(title) + `</text>`,
		`<text x="16" y="44" fill="#555">` + esc(subtitle) + `</text>`}
	for i, se := range ser {
		p := pos[i]
		s = append(s, fmt.Sprintf(`<circle cx="%s" cy="%d" r="5" fill="%s"/>`, p.x.addInt(6), 60+p.y, se.color)+
			fmt.Sprintf(`<text x="%s" y="%d" fill="#333">%s</text>`, p.x.addInt(17), 64+p.y, esc(se.name)))
	}
	bot := top + gh*len(groups)
	for _, t := range ticks(lo, hi) {
		s = append(s, fmt.Sprintf(`<line x1="%s" x2="%s" y1="%d" y2="%d" stroke="#ddd"/>`, ff(x(t), 1), ff(x(t), 1), top-6, bot))
		s = append(s, fmt.Sprintf(`<text x="%s" y="%d" text-anchor="middle" fill="#444">%s</text>`, ff(x(t), 1), bot+16, pyG(t, 6)))
	}
	s = append(s, fmt.Sprintf(`<text x="%s" y="%d" text-anchor="middle" fill="#444">%s (log axis)</text>`, pyRepr(float64(labelW)+float64(plotW)/2), bot+34, unit))
	for i, g := range groups {
		y := top + i*gh
		s = append(s, fmt.Sprintf(`<text x="%d" y="%s" text-anchor="end" fill="#222">%s</text>`, labelW-8, pyRepr(float64(y)+float64(bh*n)/2+4), esc(prefix(g.label, 46))))
		for k, vp := range g.vals {
			if vp == nil || *vp == 0 {
				continue
			}
			v := *vp
			yy := y + k*bh
			yc := float64(yy) + float64(bh)/2 - 0.5
			// dots, not bars: a bar's length on a log axis depends on an arbitrary baseline
			s = append(s, fmt.Sprintf(`<line x1="%d" x2="%s" y1="%s" y2="%s" stroke="%s" stroke-opacity="0.35"/>`, labelW, ff(x(v), 1), ff(yc, 1), ff(yc, 1), ser[k].color))
			s = append(s, fmt.Sprintf(`<circle cx="%s" cy="%s" r="4.2" fill="%s"/>`, ff(x(v), 1), ff(yc, 1), ser[k].color))
			s = append(s, fmt.Sprintf(`<text x="%s" y="%d" fill="#333" font-size="10">%s</text>`, ff(x(v)+8, 1), yy+bh-2, pyG(v, 3)))
		}
	}
	s = append(s, "</svg>")
	return strings.Join(s, "\n"), true
}

func (n pyNum) addInt(k int) string {
	if n.isInt {
		return strconv.FormatInt(int64(n.f)+int64(k), 10)
	}
	return pyRepr(n.f + float64(k))
}

type pt struct{ x, y *float64 }

type lineSeries struct {
	name, color string
	pts         []pt
}

// usable is Python's `p[0] > 0 and p[1] > 0`; comparing None raises.
func usable(p pt) (bool, error) {
	if p.x == nil {
		return false, errors.New("TypeError: '>' not supported between 'NoneType' and 'int'")
	}
	if !(*p.x > 0) {
		return false, nil
	}
	if p.y == nil {
		return false, errors.New("TypeError: '>' not supported between 'NoneType' and 'int'")
	}
	return *p.y > 0, nil
}

func lineChart(ser []lineSeries, title, subtitle, xlabel, ylabel string, logx, logy bool) (string, bool, error) {
	const w, h = 720, 380
	var xs, ys []float64
	for _, se := range ser {
		for _, p := range se.pts {
			ok, err := usable(p)
			if err != nil {
				return "", false, err
			}
			if ok {
				xs, ys = append(xs, *p.x), append(ys, *p.y)
			}
		}
	}
	if len(xs) == 0 {
		return "", false, nil
	}
	xlo, xhi, ylo, yhi := minOf(xs), maxOf(xs), minOf(ys)*0.8, maxOf(ys)*1.25
	const L, R, T, B = 70, 150, 80, 60
	fx := func(v float64) float64 { return v }
	if logx {
		fx = math.Log
	}
	fy := func(v float64) float64 { return v }
	if logy {
		fy = math.Log
	}
	orOne := func(d float64) float64 {
		if d == 0 {
			return 1
		}
		return d
	}
	px := func(v float64) float64 {
		return float64(L) + float64((fx(v)-fx(xlo))/orOne(fx(xhi)-fx(xlo))*float64(w-L-R))
	}
	py := func(v float64) float64 {
		return float64(h-B) - float64((fy(v)-fy(ylo))/orOne(fy(yhi)-fy(ylo))*float64(h-T-B))
	}
	s := []string{fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" %s>`, w, h, w, h, font),
		`<rect width="100%" height="100%" fill="#fff"/>`,
		`<text x="16" y="24" font-size="15" font-weight="bold" fill="#111">` + esc(title) + `</text>`,
		`<text x="16" y="44" fill="#555">` + esc(subtitle) + `</text>`}
	for _, t := range ticks(ylo, yhi) {
		s = append(s, fmt.Sprintf(`<line x1="%d" x2="%d" y1="%s" y2="%s" stroke="#e5e5e5"/>`, L, w-R, ff(py(t), 1), ff(py(t), 1))+
			fmt.Sprintf(`<text x="%d" y="%s" text-anchor="end" fill="#444">%s</text>`, L-6, ff(py(t)+4, 1), pyG(t, 6)))
	}
	for _, t := range ticks(xlo, xhi) {
		s = append(s, fmt.Sprintf(`<line x1="%s" x2="%s" y1="%d" y2="%d" stroke="#eee"/>`, ff(px(t), 1), ff(px(t), 1), T-6, h-B)+
			fmt.Sprintf(`<text x="%s" y="%d" text-anchor="middle" fill="#444">%s</text>`, ff(px(t), 1), h-B+16, pyG(t, 6)))
	}
	s = append(s, fmt.Sprintf(`<text x="%s" y="%d" text-anchor="middle" fill="#333">%s</text>`, pyRepr(float64(L+w-R)/2), h-12, esc(xlabel)))
	s = append(s, fmt.Sprintf(`<text x="16" y="%d" fill="#333">%s</text>`, T-14, esc(ylabel)))
	for k, se := range ser {
		var ps [][2]float64
		for _, p := range se.pts {
			ok, err := usable(p)
			if err != nil {
				return "", false, err
			}
			if ok {
				ps = append(ps, [2]float64{*p.x, *p.y})
			}
		}
		if len(ps) == 0 {
			continue
		}
		sort.SliceStable(ps, func(i, j int) bool {
			if ps[i][0] != ps[j][0] {
				return ps[i][0] < ps[j][0]
			}
			return ps[i][1] < ps[j][1]
		})
		var d []string
		for i, p := range ps {
			op := "L"
			if i == 0 {
				op = "M"
			}
			d = append(d, op+ff(px(p[0]), 1)+","+ff(py(p[1]), 1))
		}
		s = append(s, `<path d="`+strings.Join(d, " ")+`" fill="none" stroke="`+se.color+`" stroke-width="2"/>`)
		for _, p := range ps {
			s = append(s, fmt.Sprintf(`<circle cx="%s" cy="%s" r="2.6" fill="%s"/>`, ff(px(p[0]), 1), ff(py(p[1]), 1), se.color))
		}
		s = append(s, fmt.Sprintf(`<rect x="%d" y="%d" width="10" height="10" fill="%s"/>`, w-R+12, T+k*18, se.color)+
			fmt.Sprintf(`<text x="%d" y="%d" fill="#333">%s</text>`, w-R+27, T+k*18+9, esc(se.name)))
	}
	s = append(s, "</svg>")
	return strings.Join(s, "\n"), true, nil
}

var pairedRow = regexp.MustCompile(`^\|\s*(?P<row>[^|]+?)\s*\|\s*(?P<o>[\d.]+)\s*/\s*(?P<om>[\d.]+)\s*\|\s*(?P<t>[\d.]+)\s*/\s*(?P<tm>[\d.]+)\s*\|\s*(?P<r>[\d.]+)x\s*\[(?P<rlo>[\d.]+)-(?P<rhi>[\d.]+)\]\s*\|(?P<rest>.*)$`)

type missRow struct{ row, reason string }

// lanes keeps headings in first-seen order, as a Python dict does.
type lanes struct {
	order []string
	rows  map[string][]ratioRow
}

// parsePaired reads a paired GPU-vs-torch summary.md: {lane: [rows]}. ratio
// = torch/ojas, so time ratio ojas/torch = 1/ratio (lower is faster). Rows of
// a "vs" lane table that carry no ratio (refused, failed) go to missing so the
// caller can say they were not charted.
func parsePaired(text string, missing map[string][]missRow) (*lanes, error) {
	ls := &lanes{rows: map[string][]ratioRow{}}
	cur, haveCur := "", false
	idx := map[string]int{}
	for _, n := range pairedRow.SubexpNames() {
		if n != "" {
			idx[n] = pairedRow.SubexpIndex(n)
		}
	}
	for _, line := range splitLines(text) {
		if strings.HasPrefix(line, "### ") {
			cur, haveCur = pyStrip(line[4:]), true
			if _, ok := ls.rows[cur]; !ok {
				ls.order = append(ls.order, cur)
			}
			ls.rows[cur] = nil
			continue
		}
		m := pairedRow.FindStringSubmatch(line)
		if m == nil && missing != nil && haveCur && strings.Contains(cur, " vs ") &&
			strings.HasPrefix(line, "| ") && !strings.HasPrefix(line, "| row") && !strings.HasPrefix(line, "| :") {
			cells := mdCells(line)
			if len(cells) < 2 {
				return nil, errors.New("ValueError: max() arg is an empty sequence")
			}
			reason := cells[1]
			for _, c := range cells[2:] {
				if utf8.RuneCountInString(c) > utf8.RuneCountInString(reason) {
					reason = c
				}
			}
			reason = strings.TrimLeft(reason, "* ")
			missing[cur] = append(missing[cur], missRow{cells[0], prefix(reason, 110)})
		}
		if m != nil && haveCur {
			g := func(k string) (float64, error) { return pyFloat(m[idx[k]]) }
			r, err := g("r")
			if err != nil {
				return nil, err
			}
			om, err := g("om")
			if err != nil {
				return nil, err
			}
			tm, err := g("tm")
			if err != nil {
				return nil, err
			}
			rlo, err := g("rlo")
			if err != nil {
				return nil, err
			}
			rhi, err := g("rhi")
			if err != nil {
				return nil, err
			}
			row := ratioRow{name: m[idx["row"]], hollow: strings.Contains(m[idx["rest"]], "noisy"), ojasMS: om, torchMS: tm}
			switch {
			case r < 0.05 && om > 0 && tm > 0:
				// two printed decimals are too coarse here (0.01x spans 0.005x..0.015x):
				// use the ratio of the printed medians and drop the whisker
				row.v, row.mid = om/tm, fp(om/tm)
			case r > 0:
				row.v, row.mid = 1/r, fp(1/r)
				row.lo, row.hi = fp(1/r), fp(1/r)
				if rhi > 0 {
					row.lo = fp(1 / rhi)
				}
				if rlo > 0 {
					row.hi = fp(1 / rlo)
				}
			default:
				continue
			}
			ls.rows[cur] = append(ls.rows[cur], row)
		}
	}
	var keep []string
	for _, k := range ls.order {
		if len(ls.rows[k]) > 0 {
			keep = append(keep, k)
		}
	}
	ls.order = keep
	return ls, nil
}

func splitLines(text string) []string {
	if text == "" {
		return nil
	}
	return strings.Split(strings.TrimSuffix(text, "\n"), "\n")
}

// ---------------------------------------------------------------- plots: one reader per results folder
//
// Each folder keeps its own text format, so each has a small reader below. A
// reader that cannot find its data records a skip, and the page lists the
// folder as having no chart, so a missing chart is never silent.

type entry struct{ folder, file, title, note string }

type plotter struct {
	res     string            // bench/results
	files   map[string]string // svg base name -> content
	charts  []entry
	skipped [][2]string
	log     io.Writer
}

func slug(s string) string {
	return strings.Trim(regexp.MustCompile(`[^a-z0-9]+`).ReplaceAllString(strings.ToLower(s), "-"), "-")
}

func (p *plotter) add(folder, name string, ok bool, title, note string) {
	if ok {
		p.charts = append(p.charts, entry{folder, name, title, note})
		return
	}
	p.skipped = append(p.skipped, [2]string{folder, title})
	fmt.Fprintln(p.log, "SKIP", folder, title)
}

func svgName(folder, name string) string {
	return strings.ReplaceAll(folder, "/", "--") + "--" + name + ".svg"
}

func (p *plotter) put(folder, name, content string, ok bool, title, note string) {
	if ok {
		p.files[svgName(folder, name)] = content
	}
	p.add(folder, "../site/assets/plots/bench/"+svgName(folder, name), ok, title, note)
}

var numRe = regexp.MustCompile(`-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?`)

// num is the first number in a table cell, or nil.
func num(s string) *float64 {
	m := numRe.FindString(strings.ReplaceAll(strings.ReplaceAll(s, ",", ""), "−", "-"))
	if m == "" {
		return nil
	}
	f, err := pyFloat(m)
	if err != nil {
		return nil
	}
	return &f
}

func numS(s string) *float64 { return num(s) }

type mdTable struct {
	header []string
	rows   [][]string
}

var sepCell = regexp.MustCompile(`^:?-{2,}:?$`)

func mdCells(line string) []string {
	parts := strings.Split(strings.Trim(pyStrip(line), "|"), "|")
	for i := range parts {
		parts[i] = pyStrip(parts[i])
	}
	return parts
}

func mdTablesText(text string) []mdTable {
	var tabs [][][]string
	var cur [][]string
	for _, line := range splitLines(text) {
		if strings.HasPrefix(line, "|") {
			cells := mdCells(line)
			all, any := true, false
			for _, c := range cells {
				if !(sepCell.MatchString(c) || c == "") {
					all = false
				}
				if c != "" {
					any = true
				}
			}
			if all && any {
				continue
			}
			cur = append(cur, cells)
		} else if len(cur) > 0 {
			tabs = append(tabs, cur)
			cur = nil
		}
	}
	if len(cur) > 0 {
		tabs = append(tabs, cur)
	}
	var out []mdTable
	for _, t := range tabs {
		if len(t) > 1 {
			out = append(out, mdTable{t[0], t[1:]})
		}
	}
	return out
}

func mdTables(path string) ([]mdTable, error) {
	t, err := readText(path)
	if err != nil {
		return nil, err
	}
	return mdTablesText(t), nil
}

// cell is r[i], refusing a short row as Python's IndexError would.
func cell(r []string, i int) (string, error) {
	if i >= len(r) {
		return "", fmt.Errorf("IndexError: column %d of %q", i, r)
	}
	return r[i], nil
}

func tableAt(tabs []mdTable, i int, what string) (mdTable, error) {
	if i >= len(tabs) {
		return mdTable{}, fmt.Errorf("IndexError: no table %d in %s", i, what)
	}
	return tabs[i], nil
}

func firstTable(tabs []mdTable, pred func(mdTable) bool) (mdTable, bool) {
	for _, t := range tabs {
		if pred(t) {
			return t, true
		}
	}
	return mdTable{}, false
}

func headerIs(h string) func(mdTable) bool {
	return func(t mdTable) bool { return len(t.header) > 0 && t.header[0] == h }
}

func (p *plotter) emitRatio(folder, name string, rows []ratioRow, title, subtitle, note string, band float64, sorted bool) {
	if sorted {
		rows = append([]ratioRow(nil), rows...)
		sort.SliceStable(rows, func(i, j int) bool { return rows[i].v < rows[j].v })
	}
	c, ok := ratioChart(rows, title, subtitle, band)
	p.put(folder, name, c, ok, title, note)
}

func (p *plotter) emitGrouped(folder, name string, groups []group, ser []series, title, subtitle, unit, note string) {
	c, ok := groupedChart(groups, ser, title, subtitle, unit)
	p.put(folder, name, c, ok, title, note)
}

func (p *plotter) emitLines(folder, name string, ser []lineSeries, title, subtitle, xl, yl, note string, logx, logy bool) error {
	c, ok, err := lineChart(ser, title, subtitle, xl, yl, logx, logy)
	if err != nil {
		return err
	}
	p.put(folder, name, c, ok, title, note)
	return nil
}

const band = 1.1

func exists(path string) bool { _, err := os.Stat(path); return err == nil }

var roundsRe = regexp.MustCompile(`(?m)^(\d+) rounds`)

// paired: one paired GPU-vs-torch summary. text, when set, is the summary
// itself rather than the folder's summary.md.
func (p *plotter) paired(folder string, text *string) error {
	head := ""
	if text != nil {
		head = *text
	} else {
		path := filepath.Join(p.res, folder, "summary.md")
		if !exists(path) {
			return nil
		}
		t, err := readText(path)
		if err != nil {
			return err
		}
		head = t
	}
	missing := map[string][]missRow{}
	ls, err := parsePaired(head, missing)
	if err != nil {
		return err
	}
	n := "?"
	if m := roundsRe.FindStringSubmatch(head); m != nil {
		n = m[1]
	}
	for _, lane := range ls.order {
		rows := ls.rows[lane]
		miss := missing[lane]
		extra := ""
		if len(miss) > 0 {
			var parts []string
			for _, m := range miss[:min(4, len(miss))] {
				parts = append(parts, "`"+m.row+"` ("+m.reason+")")
			}
			extra = fmt.Sprintf(" %d further row(s) have no ratio and are not charted: ", len(miss)) + strings.Join(parts, "; ") + "."
		}
		p.emitRatio(folder, slug(lane), rows, folder+": "+lane,
			"ojas median / torch median, median of "+n+" per-round ratios",
			fmt.Sprintf("%s, %d rows charted, %s paired rounds. Hollow bars were flagged noisy by the run.", lane, len(rows), n)+extra, band, true)
	}
	return nil
}

var probeOld = regexp.MustCompile(`^(.*)old-(\d+)\.md$`)

// probeAB: old/new probe markdown pairs.
func (p *plotter) probeAB(folder, sub string) error {
	d := filepath.Join(p.res, folder, sub)
	type ok struct {
		pre string
		i   int
	}
	var keys []ok
	olds := map[ok]string{}
	for _, f := range globSorted(filepath.Join(d, "*old-*.md")) {
		m := probeOld.FindStringSubmatch(filepath.Base(f))
		if m == nil {
			continue
		}
		i, _ := strconv.Atoi(m[2])
		k := ok{m[1], i}
		if _, seen := olds[k]; !seen {
			keys = append(keys, k)
		}
		olds[k] = f
	}
	type pair struct{ a, b float64 }
	var labels []string
	perRound := map[string]map[int]pair{}
	roundOrder := map[string][]int{}
	for _, k := range keys {
		fo := olds[k]
		fn := filepath.Join(d, fmt.Sprintf("%snew-%d.md", k.pre, k.i))
		if !exists(fn) {
			continue
		}
		var vals [2]map[string]float64
		var order [2][]string
		for s, path := range []string{fo, fn} {
			vals[s] = map[string]float64{}
			tabs, err := mdTables(path)
			if err != nil {
				return err
			}
			for _, t := range tabs {
				col := -1
				for c, h := range t.header {
					if strings.HasPrefix(h, "min ") {
						col = c
						break
					}
				}
				if col < 0 {
					continue
				}
				for _, r := range t.rows {
					c, err := cell(r, col)
					if err != nil {
						return err
					}
					if v := num(c); v != nil && *v != 0 && *v > 0 {
						if _, seen := vals[s][r[0]]; !seen {
							order[s] = append(order[s], r[0])
						}
						vals[s][r[0]] = *v
					}
				}
			}
		}
		for _, lab := range order[0] {
			if b, ok := vals[1][lab]; ok {
				if _, seen := perRound[lab]; !seen {
					perRound[lab] = map[int]pair{}
					labels = append(labels, lab)
				}
				if _, seen := perRound[lab][k.i]; !seen {
					roundOrder[lab] = append(roundOrder[lab], k.i)
				}
				perRound[lab][k.i] = pair{vals[0][lab], b}
			}
		}
	}
	var rows []ratioRow
	for _, lab := range labels {
		var as, bs, ratios []float64
		for _, i := range roundOrder[lab] {
			pr := perRound[lab][i]
			as, bs, ratios = append(as, pr.a), append(bs, pr.b), append(ratios, pr.b/pr.a)
		}
		o, nw := minOf(as), minOf(bs)
		rows = append(rows, ratioRow{name: lab, v: nw / o, lo: fp(minOf(ratios)), hi: fp(maxOf(ratios)), mid: fp(median(ratios))})
	}
	n := len(olds)
	sfx := strings.ReplaceAll(sub, "/", "--")
	p.emitRatio(folder, sfx, rows, folder+"/"+sub+": new / old",
		fmt.Sprintf("min over %d interleaved rounds per side; whisker = per-round new/old range, circle = median", n),
		pyStrip(fmt.Sprintf("Interleaved A/B, %d rounds per side, %d rows. ", n, len(rows))), band, true)
	return nil
}

var sumRe = regexp.MustCompile(`^(\S+) (old|new) rounds=(\d+) min=(\S+) med_of_mins=(\S+) med_of_medians=(\S+)`)

// abSummary: old/new jsonl dirs via summary.txt.
func (p *plotter) abSummary(folder, sub string) error {
	path := filepath.Join(p.res, folder, sub, "summary.txt")
	if !exists(path) {
		return nil
	}
	lines, err := fileLines(path)
	if err != nil {
		return err
	}
	type side struct {
		min, medMins float64
		rounds       int
	}
	var order []string
	d := map[string]map[string]side{}
	for _, line := range lines {
		m := sumRe.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		var f [3]float64
		for i := range f {
			if f[i], err = pyFloat(m[4+i]); err != nil {
				return err
			}
		}
		r, err := strconv.Atoi(m[3])
		if err != nil {
			return err
		}
		if _, ok := d[m[1]]; !ok {
			d[m[1]] = map[string]side{}
			order = append(order, m[1])
		}
		d[m[1]][m[2]] = side{f[0], f[1], r}
	}
	var rows []ratioRow
	n := 0
	for _, row := range order {
		s := d[row]
		o, okO := s["old"]
		nw, okN := s["new"]
		if okO && okN && o.min > 0 {
			n = nw.rounds
			rows = append(rows, ratioRow{name: row, v: nw.min / o.min, mid: fp(nw.medMins / o.medMins)})
		}
	}
	name := sub
	if name == "" {
		name = "ab"
	}
	p.emitRatio(folder, strings.ReplaceAll(name, "/", "--"), rows, strings.TrimRight(folder+"/"+sub, "/")+": new / old",
		fmt.Sprintf("min-of-rounds ratio, circle = median-of-round-minimums ratio, %d rounds", n),
		fmt.Sprintf("Old binary against new binary, %d rounds, %d rows.", n, len(rows)), band, true)
	return nil
}

// tableSeries: (label, values) per row of one markdown table.
func tableSeries(path string, labelCol int, valueCols []int) ([]group, error) {
	tabs, err := mdTables(path)
	if err != nil {
		return nil, err
	}
	t, err := tableAt(tabs, 0, path)
	if err != nil {
		return nil, err
	}
	var groups []group
	for _, r := range t.rows {
		l, err := cell(r, labelCol)
		if err != nil {
			return nil, err
		}
		var vs []*float64
		for _, c := range valueCols {
			x, err := cell(r, c)
			if err != nil {
				return nil, err
			}
			vs = append(vs, num(x))
		}
		groups = append(groups, group{l, vs})
	}
	return groups, nil
}

// cols reads the given columns of a table row as numbers.
func cols(r []string, is ...int) ([]*float64, error) {
	var out []*float64
	for _, i := range is {
		c, err := cell(r, i)
		if err != nil {
			return nil, err
		}
		out = append(out, num(c))
	}
	return out, nil
}

func (p *plotter) kernelTables() error {
	f := "2026-10-02-kernels"
	for _, fn := range []string{"kernels.md", "kernels-after-fold.md"} {
		g, err := tableSeries(filepath.Join(p.res, f, fn), 0, []int{3}) // header: kernel, elements, f32 passes, min µs, median µs, GB/s
		if err != nil {
			return err
		}
		p.emitGrouped(f, slug(fn), g, []series{{"min µs (lower is faster)", blue}},
			f+"/"+fn, "Metal kernels, one dispatch, GPU timestamps", "µs",
			"Per-dispatch GPU time at the minimum. The sequence rows are the old separate finite-check passes.")
	}
	tabs, err := mdTables(filepath.Join(p.res, f, "kernels.md"))
	if err != nil {
		return err
	}
	t, err := tableAt(tabs, 0, "kernels.md")
	if err != nil {
		return err
	}
	var g []group
	for _, r := range t.rows {
		v, err := cols(r, 5)
		if err != nil {
			return err
		}
		g = append(g, group{r[0], v})
	}
	p.emitGrouped(f, "bandwidth", g, []series{{"GB/s at min (higher is faster)", green}},
		f+": achieved bandwidth, before the fold", "GB/s at the minimum", "GB/s",
		"Memory bandwidth reached by each kernel before the finite checks were folded in.")
	return nil
}

// rowIndex is {r[0]: r for r in rows}: first position, last value.
func rowIndex(rows [][]string) ([]string, map[string][]string) {
	var order []string
	m := map[string][]string{}
	for _, r := range rows {
		if _, ok := m[r[0]]; !ok {
			order = append(order, r[0])
		}
		m[r[0]] = r
	}
	return order, m
}

func (p *plotter) floorTables() error {
	f := "2026-10-01-floor"
	ta, err := mdTables(filepath.Join(p.res, f, "floor.md"))
	if err != nil {
		return err
	}
	tb, err := mdTables(filepath.Join(p.res, f, "floor-after-condvar.md"))
	if err != nil {
		return err
	}
	a, err := tableAt(ta, 0, "floor.md")
	if err != nil {
		return err
	}
	b, err := tableAt(tb, 0, "floor-after-condvar.md")
	if err != nil {
		return err
	}
	order, hb := rowIndex(a.rows)
	_, hd := rowIndex(b.rows)
	var groups []group
	for _, k := range order {
		if _, ok := hd[k]; !ok {
			continue
		}
		x, err := cols(hb[k], 4, 5)
		if err != nil {
			return err
		}
		y, err := cols(hd[k], 4, 5)
		if err != nil {
			return err
		}
		groups = append(groups, group{k, []*float64{x[0], y[0], x[1], y[1]}})
	}
	p.emitGrouped(f, "floor", groups, []series{{"p50 before", blue}, {"p50 after condvar", green},
		{"p90 before", orange}, {"p90 after condvar", red}},
		f+": submit-and-wait floor, before and after the condvar wait",
		"Per-scenario wall time over 500 runs", "ms",
		"Median and 90th-percentile wall time of tiny GPU calls. The first run was at load 45 with the GPU 100% busy.")
	return nil
}

var gemmKind = regexp.MustCompile(`^(nt|nn|tn) (.*)`)

func (p *plotter) gemmTables() error {
	f := "2026-10-02-gemm"
	for _, fn := range []string{"gemm.md", "gemm-relabelled.md"} {
		tabs, err := mdTables(filepath.Join(p.res, f, fn))
		if err != nil {
			return err
		}
		t, err := tableAt(tabs, 0, fn)
		if err != nil {
			return err
		}
		var order []string
		by := map[string]map[string]*float64{}
		for _, r := range t.rows {
			m := gemmKind.FindStringSubmatch(r[0])
			if m == nil {
				continue
			}
			key := m[2]
			if _, ok := by[key]; !ok {
				by[key] = map[string]*float64{}
				order = append(order, key)
			}
			c, err := cell(r, 6)
			if err != nil {
				return err
			}
			by[key][m[1]] = num(c)
		}
		var groups []group
		for _, k := range order {
			groups = append(groups, group{k, []*float64{by[k]["nt"], by[k]["nn"], by[k]["tn"]}})
		}
		p.emitGrouped(f, slug(fn), groups, []series{{"nt", blue}, {"nn", orange}, {"tn", green}},
			f+"/"+fn+": exact-f32 GEMM throughput", "TFLOP/s at the minimum, higher is faster", "TFLOP/s",
			"Before the panel-walk fix. The LM-head nt and nn rows drop to about 2 TFLOP/s while tn stays near 5.5.")
	}
	for _, tn := range []string{"tn-sweep-1", "tn-sweep-2"} {
		if err := p.tnSweep(tn); err != nil {
			return err
		}
	}
	g1, err := tableSeries(filepath.Join(p.res, "2026-10-02-gate", "gate-probe-1.md"), 0, []int{2})
	if err != nil {
		return err
	}
	g2, err := tableSeries(filepath.Join(p.res, "2026-10-02-gate", "gate-probe-2.md"), 0, []int{2})
	if err != nil {
		return err
	}
	d2 := map[string][]*float64{}
	for _, g := range g2 {
		d2[g.label] = g.vals
	}
	var groups []group
	for _, g := range g1 {
		var second *float64
		if v, ok := d2[g.label]; ok && len(v) > 0 {
			second = v[0]
		}
		groups = append(groups, group{g.label, []*float64{g.vals[0], second}})
	}
	p.emitGrouped("2026-10-02-gate", "gate-probe", groups, []series{{"probe 1", blue}, {"probe 2", orange}},
		"2026-10-02-gate: gate_bwd dispatch profile before the fix", "GPU span of each dispatch, min µs", "µs",
		"ojas_per_head_gate_dbias (the serial bias sum) is the 800 µs outlier the fix removes.")
	return nil
}

// pyCapitalize is str.capitalize(): first letter upper, the rest lower.
func pyCapitalize(s string) string {
	r, n := utf8.DecodeRuneInString(s)
	if n == 0 {
		return s
	}
	return string(unicode.ToTitle(r)) + strings.ToLower(s[n:])
}

func (p *plotter) tnSweep(tn string) error {
	tabs, err := mdTables(filepath.Join(p.res, "2026-10-02-gate", "tn-sweep", tn+".md"))
	if err != nil {
		return err
	}
	t, err := tableAt(tabs, 0, tn)
	if err != nil {
		return err
	}
	type v3 struct{ single, routed, par *float64 }
	var order []string
	by := map[string]*v3{}
	for _, r := range t.rows {
		k := r[0]
		if _, ok := by[k]; !ok {
			by[k] = &v3{}
			order = append(order, k)
		}
		c3, err := cell(r, 3)
		if err != nil {
			return err
		}
		v := num(c3)
		c2, err := cell(r, 2)
		if err != nil {
			return err
		}
		switch {
		case strings.HasPrefix(c2, "single"):
			by[k].single = v
		case strings.HasPrefix(c2, "routed"):
			by[k].routed = v
		case strings.HasPrefix(c2, "parallel"):
			if by[k].par == nil {
				by[k].par = v
			} else {
				if v == nil {
					return errors.New("TypeError: '<' not supported between 'NoneType' and 'float'")
				}
				by[k].par = fp(math.Min(*by[k].par, *v))
			}
		}
	}
	var groups []group
	for _, k := range order {
		groups = append(groups, group{k, []*float64{by[k].single, by[k].routed, by[k].par}})
	}
	quiet := "contended (other jobs running; the README says -2 supersedes it)"
	if strings.HasSuffix(tn, "2") {
		quiet = "quiet slot"
	}
	p.emitGrouped("2026-10-02-gate", tn, groups,
		[]series{{"single dispatch", red}, {"routed", blue}, {"best parallel split-K", green}},
		"2026-10-02-gate/"+tn+": TN GEMM, single dispatch against split-K", "min µs per shape (M, N, K), lower is faster",
		"µs", pyCapitalize(quiet)+". Split-K at width 128 to 2048 against the single dispatch and what tessl routes to.")
	return nil
}

var pairNums = regexp.MustCompile(`\d+(?:\.\d+)?`)

func firstTwo(s string) ([]float64, error) {
	var out []float64
	for _, m := range pairNums.FindAllString(strings.ReplaceAll(s, ",", ""), 2) {
		f, err := pyFloat(m)
		if err != nil {
			return nil, err
		}
		out = append(out, f)
	}
	if len(out) < 2 {
		return nil, fmt.Errorf("IndexError: fewer than two numbers in %q", s)
	}
	return out, nil
}

func (p *plotter) lappi() error {
	for _, f := range []string{"2026-10-03-lappi-inference", "2026-10-03-lappi-inference-rerun"} {
		tabs, err := mdTables(filepath.Join(p.res, f, "README.md"))
		if err != nil {
			return err
		}
		t, ok := firstTable(tabs, headerIs("Task"))
		if !ok {
			p.skipped = append(p.skipped, [2]string{f, "lappi table"})
			continue
		}
		var groups, dgroups []group
		for _, r := range t.rows {
			var c [5]string
			for i, k := range []int{0, 1, 3, 4, 6} {
				if c[i], err = cell(r, k); err != nil {
					return err
				}
			}
			a, err := firstTwo(c[2])
			if err != nil {
				return err
			}
			b, err := firstTwo(c[3])
			if err != nil {
				return err
			}
			label := c[0] + " " + c[1]
			groups = append(groups, group{label, []*float64{fp(a[0]), fp(a[1]), fp(b[0]), fp(b[1])}})
			d, err := firstTwo(c[4])
			if err != nil {
				return err
			}
			dgroups = append(dgroups, group{label, []*float64{fp(d[0]), fp(d[1])}})
		}
		p.emitGrouped(f, "prefill", groups, []series{{"tessl base", blue}, {"tessl Lappi", green},
			{"PyTorch MPS base", orange}, {"PyTorch MPS Lappi", red}},
			f+": Qwen3.5-2B prefill", "Prefill time per prompt, original model (base) and Lappi fine-tune", "ms",
			"tessl against PyTorch 2.12.1 MPS bf16. ojas has no Qwen3.5 inference path, so it is not in this comparison. "+
				"The README tables were built from the raw logs in this folder; the charts read the tables.")
		p.emitGrouped(f, "decision", dgroups, []series{{"tessl base", blue}, {"tessl Lappi", green}},
			f+": tessl full decision time", "Prefill plus decision head, per prompt", "ms",
			"Base and Lappi weights run at the same speed, as expected for equal shapes.")
	}
	return nil
}

func groupsOf(rows [][]string, is ...int) ([]group, error) {
	var g []group
	for _, r := range rows {
		v, err := cols(r, is...)
		if err != nil {
			return nil, err
		}
		g = append(g, group{r[0], v})
	}
	return g, nil
}

func (p *plotter) percall() error {
	for _, ff := range [][2]string{{"2026-10-04-percall", "percall.md"}, {"2026-10-04-split", "percall.md"}} {
		f := ff[0]
		tabs, err := mdTables(filepath.Join(p.res, f, ff[1]))
		if err != nil {
			return err
		}
		if a, ok := firstTable(tabs, headerIs("scenario")); ok {
			g, err := groupsOf(a.rows, 2, 3)
			if err != nil {
				return err
			}
			p.emitGrouped(f, "percall", g, []series{{"min ms", blue}, {"p50 ms", orange}}, f+": Metal decode call cost",
				"H 12, D 64, 1024 cached positions, 500 runs", "ms",
				"One request, one batched call, and sixteen separate calls.")
		}
		if b, ok := firstTable(tabs, func(t mdTable) bool { return strings.HasPrefix(t.header[0], "command buffer holds") }); ok {
			g, err := groupsOf(b.rows, 1, 2)
			if err != nil {
				return err
			}
			p.emitGrouped(f, "gpu-span", g, []series{{"min µs", blue}, {"median µs", orange}}, f+": GPU span of one command buffer",
				"tessl directly, 500 buffers", "µs", "Batched dispatch against sixteen separate dispatches.")
		}
	}
	return nil
}

var errStop = errors.New("StopIteration")

func (p *plotter) sweepFit() error {
	f := "2026-10-04-sweep"
	tabs, err := mdTables(filepath.Join(p.res, f, "fit.md"))
	if err != nil {
		return err
	}
	t, ok := firstTable(tabs, headerIs("N"))
	if !ok {
		return errStop
	}
	names := []string{"ojas-metal", "ojas-wgpu", "torch-mps"}
	var ser []lineSeries
	for k, nm := range names {
		var pts []pt
		for _, r := range t.rows {
			c, err := cell(r, 1+k)
			if err != nil {
				return err
			}
			parts := strings.Split(c, "/")
			if len(parts) < 2 {
				return fmt.Errorf("IndexError: no '/' in %q", c)
			}
			pts = append(pts, pt{num(r[0]), num(parts[1])})
		}
		ser = append(ser, lineSeries{nm, palette[k], pts})
	}
	if err := p.emitLines(f, "silu-over-n", ser, f+": silu_forward time against size",
		"Median of per-round medians; the floor is the fixed cost of one call", "elements N", "ms",
		"Below roughly 4M elements every runtime sits on its floor.", true, true); err != nil {
		return err
	}
	d, ok := firstTable(tabs, headerIs("B"))
	if !ok {
		return errStop
	}
	ser = nil
	idx := 0
	for _, rt := range names {
		for _, mc := range []struct {
			mode string
			col  int
		}{{"batched", 3}, {"split", 5}} {
			var pts []pt
			for _, r := range d.rows {
				c1, err := cell(r, 1)
				if err != nil {
					return err
				}
				if c1 != rt {
					continue
				}
				c, err := cell(r, mc.col)
				if err != nil {
					return err
				}
				if v := num(c); v != nil {
					pts = append(pts, pt{num(r[0]), v})
				}
			}
			ser = append(ser, lineSeries{rt + " " + mc.mode, palette[idx%len(palette)], pts})
			idx++
		}
	}
	return p.emitLines(f, "decode-batching", ser, f+": decode cost per request", "B requests in one call against B calls then one sync",
		"requests B", "ms per request", "Batching amortises the per-call cost; the gap between split and batched is the extra per call.", true, true)
}

func (p *plotter) splitReadme() error {
	f := "2026-10-04-split"
	tabs, err := mdTables(filepath.Join(p.res, f, "README.md"))
	if err != nil {
		return err
	}
	for _, t := range tabs {
		if t.header[0] != "splits" {
			continue
		}
		var r []string
		for _, x := range t.rows {
			if strings.HasPrefix(x[0], "median") {
				r = x
				break
			}
		}
		if r == nil {
			return errStop
		}
		var pts []pt
		for i := 1; i < len(t.header) && i < len(r); i++ {
			pts = append(pts, pt{num(t.header[i]), num(r[i])})
		}
		return p.emitLines(f, "splits", []lineSeries{{"median µs", blue, pts}}, f+": split count for the decode attention kernel",
			"median µs against number of key splits", "splits", "µs", "Time bottoms out near eight splits.", true, false)
	}
	p.skipped = append(p.skipped, [2]string{f, "splits table"})
	return nil
}

var clipOjas = regexp.MustCompile(`ojas at ([\d.]+) ms median on Metal`)

func (p *plotter) clip() error {
	f := "2026-10-05-torch215-clip"
	path := filepath.Join(p.res, f, "README.md")
	tabs, err := mdTables(path)
	if err != nil {
		return err
	}
	t, err := tableAt(tabs, 0, path)
	if err != nil {
		return err
	}
	var g []group
	for _, r := range t.rows {
		v, err := cols(r, 1)
		if err != nil {
			return err
		}
		g = append(g, group{"torch " + r[0], v})
	}
	txt, err := readText(path)
	if err != nil {
		return err
	}
	if m := clipOjas.FindStringSubmatch(txt); m != nil {
		v, err := pyFloat(m[1])
		if err != nil {
			return err
		}
		g = append(g, group{"ojas Metal (round 5, unpaired)", []*float64{fp(v)}})
	}
	p.emitGrouped(f, "clip", g, []series{{"median ms", blue}}, f+": clip_grad_norm on MPS",
		"Median of run 1; lower is faster", "ms",
		"The folder's README attributes the 16x drop to a change in torch's MPS norm reduction and says a paired run with torch 2.15 is still needed before a new ratio is quoted.")
	return nil
}

func (p *plotter) gateSaved() error {
	f := "2026-10-07-gate-saved"
	tabs, err := mdTables(filepath.Join(p.res, f, "README.md"))
	if err != nil {
		return err
	}
	t, ok := firstTable(tabs, headerIs("Backend"))
	if !ok {
		return errStop
	}
	var rows []ratioRow
	for _, r := range t.rows {
		for _, kn := range []struct {
			k  int
			nm string
		}{{2, "forward_saving / forward"}, {3, "backward_saved / backward"}} {
			c, err := cell(r, kn.k)
			if err != nil {
				return err
			}
			parts := strings.Split(c, "/")
			if len(parts) != 2 {
				return fmt.Errorf("ValueError: %q does not split into two", c)
			}
			mn, md := num(parts[0]), num(parts[1])
			if md == nil {
				return errors.New("TypeError: '<' not supported between 'NoneType' and 'float'")
			}
			rows = append(rows, ratioRow{name: r[0] + " round " + r[1] + ": " + kn.nm, v: *md, mid: mn, hollow: strings.Contains(r[1], "noisy")})
		}
	}
	p.emitRatio(f, "gate-saved", rows, f+": saved-sigmoid gate pair against recompute",
		"saved / plain, median per round; circle = min; hollow = noisy round (started at load 24)",
		"The saved backward is 7% faster on Metal and 15% faster on wgpu; the forward costs at most about 2%.", band, false)
	return nil
}

var (
	attnRange = regexp.MustCompile(`^([\d.]+) \(([\d.]+)-([\d.]+)\)`)
	attnCtl   = regexp.MustCompile(`^([\d.]+)`)
)

func (p *plotter) attnLSE() error {
	f := "2026-10-08-attn-lse-ab"
	tabs, err := mdTables(filepath.Join(p.res, f, "tables.md"))
	if err != nil {
		return err
	}
	t, err := tableAt(tabs, 0, "tables.md")
	if err != nil {
		return err
	}
	var rows []ratioRow
	for _, r := range t.rows {
		c5, err := cell(r, 5)
		if err != nil {
			return err
		}
		c6, err := cell(r, 6)
		if err != nil {
			return err
		}
		m, c := attnRange.FindStringSubmatch(c5), attnCtl.FindStringSubmatch(c6)
		if m == nil || c == nil {
			return errors.New("TypeError: 'NoneType' object is not subscriptable")
		}
		var v [4]float64
		for i, s := range []string{m[1], m[2], m[3], c[1]} {
			if v[i], err = pyFloat(s); err != nil {
				return err
			}
		}
		rows = append(rows, ratioRow{name: r[0] + " " + r[1] + " " + r[2], v: v[0], lo: fp(v[1]), hi: fp(v[2]), mid: fp(v[3])})
	}
	p.emitRatio(f, "new-over-old", rows, f+": attention backward with saved log-sum-exp",
		"new / old per round, median and range; circle = control op (should be 1.0)",
		"Backward is 0.76 to 0.81 of the old time on every shape and both backends. Forward is within noise on the multi-head shapes; "+
			"the grouped-query shape's Metal forward is 0.89.", band, true)
	w, err := tableAt(tabs, 1, "tables.md")
	if err != nil {
		return err
	}
	var order []string
	by := map[string]map[string][2]*float64{}
	for _, r := range w.rows {
		v, err := cols(r, 3, 4)
		if err != nil {
			return err
		}
		k := r[0] + " " + r[1]
		if _, ok := by[k]; !ok {
			by[k] = map[string][2]*float64{}
			order = append(order, k)
		}
		by[k][r[2]] = [2]*float64{v[0], v[1]}
	}
	var groups []group
	for _, k := range order {
		full, ok1 := by[k]["full"]
		w256, ok2 := by[k]["w256"]
		if !ok1 || !ok2 {
			return fmt.Errorf("KeyError: %s lacks full or w256", k)
		}
		groups = append(groups, group{k, []*float64{full[0], w256[0], full[1], w256[1]}})
	}
	p.emitGrouped(f, "window", groups, []series{{"bwd full", blue}, {"bwd window 256", orange},
		{"fwd full", green}, {"fwd window 256", red}},
		f+": sliding window of 256 against full attention", "best min ms over rounds", "ms",
		"The 256-wide window cuts backward time by roughly 3x at 2048 tokens.")
	return nil
}

func (p *plotter) cpuHotPaths() error {
	f := "2026-10-08-cpu-hot-paths"
	lines, err := fileLines(filepath.Join(p.res, f, "summary.md"))
	if err != nil {
		return err
	}
	title := "ab"
	var order []string
	secs := map[string][][]string{}
	for _, line := range lines {
		if strings.HasPrefix(line, "## ") {
			title = pyStrip(line[3:])
			continue
		}
		if !strings.HasPrefix(line, "|") || strings.HasPrefix(line, "| :") || strings.HasPrefix(line, "| row") {
			continue
		}
		c := mdCells(line)
		if len(c) < 8 {
			continue
		}
		if _, ok := secs[title]; !ok {
			order = append(order, title)
		}
		secs[title] = append(secs[title], c)
	}
	for k, title := range order {
		rows := secs[title]
		tag := prefix(slug(title), 24)
		if k < 3 {
			tag = []string{"ab", "ab-tape", "ab-v2"}[k]
		}
		var rr []ratioRow
		for _, c := range rows {
			nm := c[0]
			if c[1] != "step" {
				nm += " " + c[1]
			}
			nm = strings.ReplaceAll(nm, "tape ", "")
			if v := num(c[4]); v != nil {
				rr = append(rr, ratioRow{name: nm, v: *v, mid: num(c[5])})
			}
		}
		shown := title
		if title == "ab" {
			shown = "interleaved A/B, base against after"
		}
		p.emitRatio(f, tag+"-vs-base", rr, f+": "+shown,
			"after / base, min over rounds; circle = median of per-round ratios; load 17-82 on the first table",
			fmt.Sprintf("%d rows. Ratios between 0.9 and 1.1 are noise on this machine.", len(rr)), band, true)
		var tr []ratioRow
		for _, c := range rows {
			if v := num(c[7]); v != nil {
				tr = append(tr, ratioRow{name: c[0] + " " + c[1], v: *v})
			}
		}
		if len(tr) > 0 {
			p.emitRatio(f, tag+"-vs-torch", tr, f+": ojas / torch 2.13 CPU", "min over rounds, rows torch covers",
				fmt.Sprintf("%d rows with a torch counterpart.", len(tr)), band, true)
		}
	}
	return nil
}

func (p *plotter) floorAB() error {
	f := "2026-10-01-floor"
	tabs, err := mdTables(filepath.Join(p.res, f, "ab", "summary.md"))
	if err != nil {
		return err
	}
	t, err := tableAt(tabs, 0, "ab/summary.md")
	if err != nil {
		return err
	}
	var order []string
	by := map[string]map[string][]*float64{}
	for _, r := range t.rows {
		v, err := cols(r, 4, 5)
		if err != nil {
			return err
		}
		if _, ok := by[r[0]]; !ok {
			by[r[0]] = map[string][]*float64{}
			order = append(order, r[0])
		}
		by[r[0]][r[1]] = v
	}
	var groups []group
	for _, k := range order {
		o, ok1 := by[k]["old"]
		n, ok2 := by[k]["new"]
		if ok1 && ok2 {
			groups = append(groups, group{k, []*float64{o[0], n[0], o[1], n[1]}})
		}
	}
	p.emitGrouped(f, "ab", groups, []series{{"p50 old", blue}, {"p50 new", green}, {"p90 old", orange}, {"p90 new", red}},
		f+"/ab: submit-and-wait floor, old against new", "median over rounds of the per-round p50 and p90", "ms",
		"600 runs per side. The old side had 388 to 507 slow (>1 ms) runs of 600 per scenario; the new side had 0 to 6.")
	return nil
}

var (
	kvWide = regexp.MustCompile(`(\w+)=([^\s]+)`)
	kvRe   = regexp.MustCompile(`(\w+)=(\S+)`)
)

func kvOf(re *regexp.Regexp, line string) map[string]string {
	m := map[string]string{}
	for _, g := range re.FindAllStringSubmatch(line, -1) {
		m[g[1]] = g[2]
	}
	return m
}

// kvLines yields the key=value pairs of each line starting with prefix.
func kvLines(path, pre string) ([]map[string]string, error) {
	lines, err := fileLines(path)
	if err != nil {
		return nil, err
	}
	var out []map[string]string
	for _, l := range lines {
		if strings.HasPrefix(l, pre) {
			out = append(out, kvOf(kvWide, l))
		}
	}
	return out, nil
}

func kvFloat(kv map[string]string, k string) (float64, error) {
	v, ok := kv[k]
	if !ok {
		return 0, fmt.Errorf("KeyError: %q", k)
	}
	return pyFloat(v)
}

// minTable keeps the minimum per (outer, inner) key, outer keys and inner keys
// in first-insertion order.
type minTable struct {
	order []string
	m     map[string]map[string]float64
}

func newMinTable() *minTable { return &minTable{m: map[string]map[string]float64{}} }

func (t *minTable) put(outer, inner string, v float64) {
	if _, ok := t.m[outer]; !ok {
		t.m[outer] = map[string]float64{}
		t.order = append(t.order, outer)
	}
	if old, ok := t.m[outer][inner]; !ok || v < old {
		if ok || v < 1e18 {
			t.m[outer][inner] = v
		} else {
			t.m[outer][inner] = 1e18
		}
	}
}

var gateRound = regexp.MustCompile(`^round=\d+ gate=(\S+)`)

func (p *plotter) cpuProbes() error {
	f := "2026-10-08-cpu-hot-paths"
	d := filepath.Join(p.res, f)
	if _, err := kvLines(filepath.Join(d, "muon-gate.txt"), "OJAS_OP"); err != nil {
		return err
	}
	type og struct{ op, gate string }
	var bestOrder []og
	best := map[og]float64{}
	gate := ""
	lines, err := fileLines(filepath.Join(d, "muon-gate.txt"))
	if err != nil {
		return err
	}
	for _, line := range lines {
		if m := gateRound.FindStringSubmatch(line); m != nil {
			gate = m[1]
		} else if strings.HasPrefix(line, "OJAS_OP") && gate != "" {
			kv := kvOf(kvRe, line)
			op, ok := kv["op"]
			if !ok {
				return errors.New("KeyError: 'op'")
			}
			v, err := kvFloat(kv, "min_ms")
			if err != nil {
				return err
			}
			k := og{op, gate}
			old, seen := best[k]
			if !seen {
				bestOrder = append(bestOrder, k)
				old = 1e18
			}
			best[k] = math.Min(old, v)
		}
	}
	var rows []ratioRow
	for _, k := range bestOrder {
		if b, ok := best[og{k.op, "nanolab"}]; k.gate != "nanolab" && ok {
			rows = append(rows, ratioRow{name: k.op + ": gate " + k.gate, v: best[k] / b})
		}
	}
	p.emitRatio(f, "muon-gate", rows, f+": Muon no-transpose view gate",
		"min over 3 rounds, divided by the shipped gate (nanolab); above 1 is slower than shipped",
		"Opening the view to every tall matrix (tall, tallbands) reads 1.3 to 2.9x slower than shipped, so the gate stays. "+
			"Gate off reads 0.78 to 1.21 and is noise-bound at load 20-35 (the README calls 2048x768 on versus off within noise).", band, true)

	collect := func(file, pre string, key func(map[string]string) (string, error), fields []string) (*minTable, error) {
		kvs, err := kvLines(filepath.Join(d, file), pre)
		if err != nil {
			return nil, err
		}
		t := newMinTable()
		for _, kv := range kvs {
			k, err := key(kv)
			if err != nil {
				return nil, err
			}
			for _, fld := range fields {
				v, err := kvFloat(kv, fld)
				if err != nil {
					return nil, err
				}
				t.put(k, fld, v)
			}
		}
		return t, nil
	}
	field := func(kv map[string]string, k string) (string, error) {
		v, ok := kv[k]
		if !ok {
			return "", fmt.Errorf("KeyError: %q", k)
		}
		return v, nil
	}
	syrk, err := collect(filepath.Join("probes", "syrk.txt"), "XXt", func(kv map[string]string) (string, error) {
		r, err := field(kv, "r")
		if err != nil {
			return "", err
		}
		c, err := field(kv, "c")
		if err != nil {
			return "", err
		}
		return r + "x" + c, nil
	}, []string{"gemm", "gemm_2band", "syrk_mirror"})
	if err != nil {
		return err
	}
	var g []group
	for _, k := range syrk.order {
		v := syrk.m[k]
		g = append(g, group{k, []*float64{fp(v["gemm"]), fp(v["gemm_2band"]), fp(v["syrk_mirror"])}})
	}
	p.emitGrouped(f, "syrk", g,
		[]series{{"one sgemm", blue}, {"two-band split", orange}, {"ssyrk + mirror", green}},
		f+": X times X-transpose, three ways", "min over probe runs, by matrix shape (rows x cols of X)", "ms",
		"The ssyrk product matched sgemm bit for bit at every probed shape (bits_syrk_eq_gemm).")
	tiles := []string{"serial", "t8", "t16", "t32", "t64"}
	mirror, err := collect(filepath.Join("probes", "mirror.txt"), "mirror", func(kv map[string]string) (string, error) {
		return field(kv, "n")
	}, tiles)
	if err != nil {
		return err
	}
	g = nil
	for _, k := range mirror.order {
		var vs []*float64
		for _, x := range tiles {
			vs = append(vs, fp(mirror.m[k][x]))
		}
		g = append(g, group{"n = " + k, vs})
	}
	p.emitGrouped(f, "mirror", g,
		[]series{{"column walk", blue}, {"8 tiles", orange}, {"16 tiles", green}, {"32 tiles", red}, {"64 tiles", "#7b5ea7"}},
		f+": ssyrk lower-from-upper mirror", "min over probe runs", "ms", "16 x 16 tiles shipped. At n = 2048 the 32 and 64 tile sizes are slower than the plain column walk.")

	kvs, err := kvLines(filepath.Join(d, "spawn.txt"), "scope")
	if err != nil {
		return err
	}
	pts := map[string]*minTable{"min_us": newMinTable(), "med_us": newMinTable()}
	for _, kv := range kvs {
		for _, k := range []string{"min_us", "med_us"} {
			s, err := field(kv, "spawned")
			if err != nil {
				return err
			}
			x, err := strconv.Atoi(s)
			if err != nil {
				return err
			}
			v, err := kvFloat(kv, k)
			if err != nil {
				return err
			}
			pts[k].put(strconv.Itoa(x), "v", v)
		}
	}
	line := func(t *minTable) []pt {
		var out []pt
		for _, k := range t.order {
			x, _ := strconv.Atoi(k)
			out = append(out, pt{fp(float64(x)), fp(t.m[k]["v"])})
		}
		return out
	}
	if len(pts["min_us"].order) == 0 {
		return errors.New("KeyError: 'min_us'")
	}
	if err := p.emitLines(f, "spawn", []lineSeries{{"min µs", blue, line(pts["min_us"])}, {"median µs", orange, line(pts["med_us"])}},
		f+": cost of one std::thread::scope", "min over two runs of the best and median over 2000 spawns, load 26", "threads spawned", "µs",
		"About 9 µs for one thread and 34-38 µs for five at the minimum.", false, false); err != nil {
		return err
	}

	tl, err := fileLines(filepath.Join(d, "tape-peak.txt"))
	if err != nil {
		return err
	}
	type peak struct{ ms, bytes float64 }
	var baseOrder []string
	base, v2 := map[string]peak{}, map[string]peak{}
	for _, l := range tl {
		kv := kvOf(kvRe, l)
		row, err := field(kv, "row")
		if err != nil {
			return err
		}
		ms, err := kvFloat(kv, "min_ms")
		if err != nil {
			return err
		}
		by, err := kvFloat(kv, "walk_peak_bytes")
		if err != nil {
			return err
		}
		if strings.HasPrefix(l, "base") {
			if _, ok := base[row]; !ok {
				baseOrder = append(baseOrder, row)
			}
			base[row] = peak{ms, by}
		} else {
			v2[row] = peak{ms, by}
		}
	}
	var gb, gt []group
	for _, k := range baseOrder {
		n, ok := v2[k]
		if !ok {
			return fmt.Errorf("KeyError: %q", k)
		}
		gb = append(gb, group{k, []*float64{fp(base[k].bytes), fp(n.bytes)}})
		gt = append(gt, group{k, []*float64{fp(base[k].ms), fp(n.ms)}})
	}
	p.emitGrouped(f, "tape-peak-bytes", gb, []series{{"base", blue}, {"final tree", green}},
		f+": tape walk peak charge", "bytes, from the budget counter (does not depend on machine load)", "bytes",
		"Fan-in peaks at 18,874,368 then 12,582,912 bytes. The seed one-quarter fused-CE walk falls from 154,533,892 to 4 bytes.")
	p.emitGrouped(f, "tape-peak-time", gt, []series{{"base", blue}, {"final tree", green}},
		f+": tape walk time in the peak run", "min ms of one run each, so noisy (the fan-in row reads slower here)", "ms",
		"One run per side under load; use the interleaved ab-tape table for time, not this chart.")
	return nil
}

var gqaScratch = regexp.MustCompile(`(metal|wgpu) qwen35.*forward (\d+) B \(expand path (\d+) B\), backward (\d+) B \(expand path (\d+) B\)`)

func (p *plotter) attnScratch() error {
	f := "2026-10-08-attn-lse-ab"
	lines, err := fileLines(filepath.Join(p.res, f, "gqa_scratch.txt"))
	if err != nil {
		return err
	}
	var groups []group
	for _, line := range lines {
		m := gqaScratch.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		var v [4]float64
		for i := range v {
			if v[i], err = pyFloat(m[2+i]); err != nil {
				return err
			}
		}
		groups = append(groups, group{m[1] + " forward", []*float64{fp(v[0]), fp(v[1])}}, group{m[1] + " backward", []*float64{fp(v[2]), fp(v[3])}})
	}
	p.emitGrouped(f, "gqa-scratch", groups, []series{{"native grouped-query", green}, {"expand path", red}},
		f+": scratch charged at the Qwen3.5 shape", "bytes charged by the backend's budget, b1 h8/2 t2048 d256", "bytes",
		"Reading each KV head in place cuts the charged scratch to about a third of the expanded-heads path (forward) and 27% (backward).")
	return nil
}

func (p *plotter) decodeTokens() error {
	f := "2026-10-08-decode-before"
	tabs, err := mdTables(filepath.Join(p.res, f, "summary.md"))
	if err != nil {
		return err
	}
	t, ok := firstTable(tabs, headerIs("runtime"))
	if !ok {
		return errStop
	}
	g, err := groupsOf(t.rows, 1, 2)
	if err != nil {
		return err
	}
	p.emitGrouped(f, "decode-tokens", g,
		[]series{{"prefill median ms (32 tokens)", blue}, {"decode ms per token", orange}},
		f+": generation, 32-token prompt and 32 new tokens", "3 rounds; torch-mps against each ojas lane", "ms",
		"Every ojas lane produced the same 32 greedy ids as torch. ojas-cpu decodes at 5.09 ms per token against torch-mps at 5.88; the GPU lanes are slower.")
	return nil
}

var (
	ab2Row    = regexp.MustCompile(`^(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$`)
	sweepCase = regexp.MustCompile(`^(\S+)\s+M=`)
	sweepRow  = regexp.MustCompile(`^\s+(\S+)\s+n=\d+\s+min\s+([\d.]+)\s+med\s+([\d.]+)\s+max\s+([\d.]+)`)
	tuneCase  = regexp.MustCompile(`^(\S+)\s+M=\d+`)
	tuneRow   = regexp.MustCompile(`^\s+(\S+)\s+([\d.]+)\s+\d+\s+[\d.]+×`)
)

func floats(ss ...string) ([]float64, error) {
	out := make([]float64, len(ss))
	for i, s := range ss {
		f, err := pyFloat(s)
		if err != nil {
			return nil, err
		}
		out[i] = f
	}
	return out, nil
}

func (p *plotter) gemmBF16() error {
	f := "2026-10-06-gemm-bf16"
	lines, err := fileLines(filepath.Join(p.res, f, "ab2", "summary.txt"))
	if err != nil {
		return err
	}
	var rows []ratioRow
	for _, line := range lines {
		if m := ab2Row.FindStringSubmatch(line); m != nil {
			v, err := floats(m[6], m[5], m[7])
			if err != nil {
				return err
			}
			rows = append(rows, ratioRow{name: m[1], v: v[0], mid: fp(v[0]), lo: fp(v[1]), hi: fp(v[2])})
		}
	}
	p.emitRatio(f, "ab2", rows, f+": column-panel tile walk, new / old",
		"median of 6 back-to-back pairs per case; whisker = min..max", "The largest-B cases fall to 0.5 to 0.8 of the old time; the Morton-order and small-B controls stay near 1.0.", band, true)
	lines, err = fileLines(filepath.Join(p.res, f, "sweep", "ratios.txt"))
	if err != nil {
		return err
	}
	rows = nil
	cs := ""
	for _, line := range lines {
		if h := sweepCase.FindStringSubmatch(line); h != nil {
			cs = h[1]
			continue
		}
		if m := sweepRow.FindStringSubmatch(line); m != nil && cs != "" {
			nm := strings.ReplaceAll(m[1], "mm_bf16_nt_coop_128x64_sg4", "base")
			v, err := floats(m[3], m[2], m[4])
			if err != nil {
				return err
			}
			rows = append(rows, ratioRow{name: cs + " / " + nm, v: v[0], mid: fp(v[0]), lo: fp(v[1]), hi: fp(v[2])})
		}
	}
	p.emitRatio(f, "sweep", rows, f+": panel-height sweep", "ratio to the same round's production kernel; whisker = min..max",
		"Panel heights 4, 8 and 16 against the shipped kernel across six B sizes.", band, false)
	return nil
}

func (p *plotter) gemmBF16Tune() error {
	f := "2026-10-06-gemm-bf16"
	for _, tune := range []string{"tune1", "tune2"} {
		type cv struct{ c, v string }
		times := map[cv][]float64{}
		var cases, variants []string
		seenC, seenV := map[string]bool{}, map[string]bool{}
		for _, path := range globSorted(filepath.Join(p.res, f, "sweep", tune+"-r*.txt")) {
			lines, err := fileLines(path)
			if err != nil {
				return err
			}
			cs := ""
			for _, line := range lines {
				if h := tuneCase.FindStringSubmatch(line); h != nil {
					cs = h[1]
					if !seenC[cs] {
						seenC[cs] = true
						cases = append(cases, cs)
					}
					continue
				}
				if m := tuneRow.FindStringSubmatch(line); m != nil && cs != "" {
					v := strings.ReplaceAll(m[1], "mm_bf16_", "")
					if !seenV[v] {
						seenV[v] = true
						variants = append(variants, v)
					}
					x, err := pyFloat(m[2])
					if err != nil {
						return err
					}
					times[cv{cs, v}] = append(times[cv{cs, v}], x)
				}
			}
		}
		var groups []group
		for _, c := range cases {
			var vs []*float64
			for _, v := range variants {
				if t, ok := times[cv{c, v}]; ok {
					vs = append(vs, fp(median(t)))
				} else {
					vs = append(vs, nil)
				}
			}
			groups = append(groups, group{c, vs})
		}
		var ser []series
		for i, v := range variants {
			ser = append(ser, series{v, palette[i%len(palette)]})
		}
		p.emitGrouped(f, tune+"-raw", groups, ser,
			f+": "+tune+" raw sweep", "median over 3 rounds of the in-process time per kernel variant, by case", "ms",
			"Raw times behind sweep/ratios.txt. Variants whose name ends _phN are column-panel heights.")
	}
	return nil
}

func (p *plotter) jsonlUnpaired(folder, sub string) error {
	d := filepath.Join(p.res, folder, sub)
	type rr struct{ row, rt string }
	best := map[rr]float64{}
	var rts []string
	seen := map[string]bool{}
	for _, path := range globSorted(filepath.Join(d, "round*.jsonl")) {
		lines, err := fileLines(path)
		if err != nil {
			return err
		}
		for _, line := range lines {
			j, err := decodeRecord(line)
			if err != nil {
				continue
			}
			if j.get("status") != "ok" || !j.has("min_ms") {
				continue
			}
			row, err := j.str("row")
			if err != nil {
				return err
			}
			rt, err := j.str("runtime")
			if err != nil {
				return err
			}
			v, err := j.num("min_ms")
			if err != nil {
				return err
			}
			k := rr{row, rt}
			old, ok := best[k]
			if !ok {
				old = 1e18
			}
			best[k] = math.Min(old, v)
			if !seen[rt] {
				seen[rt] = true
				rts = append(rts, rt)
			}
		}
	}
	rowSet := map[string]bool{}
	for k := range best {
		rowSet[k.row] = true
	}
	var rows []string
	for r := range rowSet {
		rows = append(rows, r)
	}
	sort.Strings(rows)
	var groups []group
	for _, r := range rows {
		var vs []*float64
		for _, rt := range rts {
			if v, ok := best[rr{r, rt}]; ok {
				vs = append(vs, fp(v))
			} else {
				vs = append(vs, nil)
			}
		}
		groups = append(groups, group{r, vs})
	}
	if len(rts) > len(palette) {
		return errors.New("IndexError: more runtimes than palette colours")
	}
	var ser []series
	for i, rt := range rts {
		ser = append(ser, series{rt, palette[i]})
	}
	p.emitGrouped(folder, "rows", groups, ser, folder+": min time per row",
		"min over rounds of each round's minimum (unpaired, machine under heavy load)", "ms",
		"Direction only. These rounds ran at load around 350.")
	return nil
}

func (p *plotter) r3() error {
	d := filepath.Join(p.res, "2026-10-01-r3")
	text, _, err := aggregateRun(d)
	if err != nil {
		p.skipped = append(p.skipped, [2]string{"2026-10-01-r3", "aggregate failed: " + err.Error()})
		return nil
	}
	return p.paired("2026-10-01-r3", &text)
}

// generate runs every reader. Errors in the per-folder readers at the end are
// recorded as skips, as one bad folder must not hide the rest; errors in the
// paired and A/B readers stop the run.
func (p *plotter) generate() error {
	for _, folder := range []string{"2026-10-01-r2", "2026-10-01-r3b", "2026-10-01-r4-smallops", "2026-10-02-r5",
		"2026-10-04-sweep", "2026-10-07-muon-bf16", "2026-10-08-decode-before",
		"2026-10-02-gate/paired-gate", "2026-10-02-gate/paired-gate-quiet"} {
		if err := p.paired(folder, nil); err != nil {
			return fmt.Errorf("%s: %v", folder, err)
		}
	}
	if err := p.r3(); err != nil {
		return err
	}
	for _, run := range []string{"runA", "runB", "runC"} {
		if err := p.paired("2026-10-01/"+run, nil); err != nil {
			return err
		}
	}
	steps := []struct {
		name string
		fn   func() error
	}{
		{"ab_summary 2026-10-01-ab", func() error { return p.abSummary("2026-10-01-ab", "") }},
	}
	for _, sub := range []string{"ab-dbias", "ab-fold", "ab-tn", "ab-stage"} {
		steps = append(steps, struct {
			name string
			fn   func() error
		}{"probe_ab 2026-10-02-gate/" + sub, func() error { return p.probeAB("2026-10-02-gate", sub) }})
	}
	for _, sub := range []string{"ab-grid-gate", "ab-footprint-gate", "ab-nn-splitk"} {
		steps = append(steps, struct {
			name string
			fn   func() error
		}{"probe_ab 2026-10-02-gemm/" + sub, func() error { return p.probeAB("2026-10-02-gemm", sub) }})
	}
	for _, sub := range []string{"ab-rows", "ab-rows-ce"} {
		steps = append(steps, struct {
			name string
			fn   func() error
		}{"ab_summary 2026-10-02-gemm/" + sub, func() error { return p.abSummary("2026-10-02-gemm", sub) }})
	}
	steps = append(steps, struct {
		name string
		fn   func() error
	}{"jsonl_unpaired", func() error { return p.jsonlUnpaired("2026-10-02-gemm", "splitk-rows") }})
	for _, sub := range []string{"ab-check", "ab-fold"} {
		steps = append(steps, struct {
			name string
			fn   func() error
		}{"ab_summary 2026-10-02-kernels/" + sub, func() error { return p.abSummary("2026-10-02-kernels", sub) }})
	}
	steps = append(steps, struct {
		name string
		fn   func() error
	}{"ab_summary 2026-10-04-split/ab", func() error { return p.abSummary("2026-10-04-split", "ab") }})
	for _, s := range steps {
		if err := s.fn(); err != nil {
			return fmt.Errorf("%s: %v", s.name, err)
		}
	}
	readers := []struct {
		name string
		fn   func() error
	}{
		{"kernel_tables", p.kernelTables}, {"floor_tables", p.floorTables}, {"gemm_tables", p.gemmTables},
		{"lappi", p.lappi}, {"percall", p.percall}, {"sweep_fit", p.sweepFit}, {"split_readme", p.splitReadme},
		{"clip", p.clip}, {"gate_saved", p.gateSaved}, {"attn_lse", p.attnLSE}, {"cpu_hot_paths", p.cpuHotPaths},
		{"gemm_bf16", p.gemmBF16}, {"gemm_bf16_tune", p.gemmBF16Tune}, {"floor_ab", p.floorAB},
		{"cpu_probes", p.cpuProbes}, {"attn_scratch", p.attnScratch}, {"decode_tokens", p.decodeTokens},
	}
	for _, r := range readers {
		if err := r.fn(); err != nil {
			p.skipped = append(p.skipped, [2]string{r.name, err.Error()})
			fmt.Fprintln(p.log, "ERROR", r.name, err)
		}
	}
	return nil
}

const docIntro = "Each chart is drawn from the text files kept under `bench/results/<folder>`, by `scripts/benchreport` (Go, standard library only). " +
	"Regenerate with `go run scripts/benchreport/main.go plots` from the repository root. Ratio charts are times, so " +
	"**lower is faster** and 1x is parity. Green bars are faster by more than 10%, red are slower by more than 10%, " +
	"grey is inside the 10% band, which is the noise this machine showed. A hollow bar means the run itself " +
	"flagged the row noisy. Whiskers are the per-round range and circles are medians where the source gives them."

// doc renders docs/bench-plots.md.
func (p *plotter) doc() (string, error) {
	by := map[string][]entry{}
	for _, m := range p.charts {
		top := strings.SplitN(m.folder, "/", 2)[0]
		by[top] = append(by[top], m)
	}
	ents, err := os.ReadDir(p.res)
	if err != nil {
		return "", err
	}
	var folders []string
	for _, e := range ents {
		if st, err := os.Stat(filepath.Join(p.res, e.Name())); err == nil && st.IsDir() {
			folders = append(folders, e.Name())
		}
	}
	sort.Strings(folders)
	L := []string{"# Benchmark Plots, Every Result Folder", "", docIntro,
		"", "Two cautions on reading them. The paired GPU summaries print ratios to two decimals, so where the " +
			"printed ratio is under 0.05x (ojas more than 20x slower) the chart uses the ratio of the two printed " +
			"medians instead and draws no whisker. That is a different statistic from the median of per-round " +
			"ratios, and on rows the run flagged noisy the two can differ by 2x or more. Dot charts of absolute " +
			"times use a log axis, so distance between dots is a ratio, not a difference.",
		"", "Read each folder's own README or summary before quoting a number: several runs were taken under heavy " +
			"load, and the charts show direction, not a verdict. The older prose analysis is in " +
			"[bench-gpu-vs-torch.md](bench-gpu-vs-torch.md) and [bench-cpu-vs-torch.md](bench-cpu-vs-torch.md).", ""}
	for _, d := range folders {
		L = append(L, "## "+d, "")
		items := by[d]
		if len(items) == 0 {
			L = append(L, "No chart: the folder holds only scripts, logs or text that is not a measurement table.", "")
			continue
		}
		for _, m := range items {
			L = append(L, "### "+m.title, "", "!["+m.title+": "+m.note+"]("+m.file+")", "", "*"+m.note+"*", "")
		}
		L = append(L, "Source: `bench/results/"+d+"/`.", "")
	}
	return strings.Join(L, "\n"), nil
}

func runPlots(root string, check bool, stdout, stderr io.Writer) int {
	res := filepath.Join(root, "bench", "results")
	out := filepath.Join(root, "site", "assets", "plots", "bench")
	docPath := filepath.Join(root, "docs", "bench-plots.md")
	if st, err := os.Stat(res); err != nil || !st.IsDir() {
		fmt.Fprintf(stderr, "plots: %s is not a directory; run from the repository root or pass -root\n", res)
		return 2
	}
	p := &plotter{res: res, files: map[string]string{}, log: stdout}
	if err := p.generate(); err != nil {
		fmt.Fprintln(stderr, "plots:", err)
		return 1
	}
	doc, err := p.doc()
	if err != nil {
		fmt.Fprintln(stderr, "plots:", err)
		return 1
	}
	var existing []string
	for _, f := range globSorted(filepath.Join(out, "*.svg")) {
		existing = append(existing, filepath.Base(f))
	}
	if check {
		var stale []string
		for name, c := range p.files {
			b, err := os.ReadFile(filepath.Join(out, name))
			if err != nil || !bytes.Equal(b, []byte(c)) {
				stale = append(stale, filepath.Join(out, name))
			}
		}
		for _, name := range existing {
			if _, ok := p.files[name]; !ok {
				stale = append(stale, filepath.Join(out, name)+" (orphan)")
			}
		}
		if b, err := os.ReadFile(docPath); err != nil || !bytes.Equal(b, []byte(doc)) {
			stale = append(stale, docPath)
		}
		sort.Strings(stale)
		for _, s := range stale {
			fmt.Fprintln(stderr, "stale:", s)
		}
		if len(stale) > 0 {
			fmt.Fprintf(stderr, "plots: %d file(s) stale; run go run scripts/benchreport/main.go plots\n", len(stale))
			return 1
		}
		if len(p.skipped) > 0 {
			fmt.Fprintf(stderr, "plots: %d chart(s) skipped\n", len(p.skipped))
			for _, s := range p.skipped {
				fmt.Fprintf(stderr, "  skipped: %s: %s\n", s[0], s[1])
			}
			return 1
		}
		fmt.Fprintf(stdout, "plots: %d charts and %s up to date\n", len(p.charts), docPath)
		return 0
	}
	// no orphans from renamed charts
	for _, name := range existing {
		if err := os.Remove(filepath.Join(out, name)); err != nil {
			fmt.Fprintln(stderr, "plots:", err)
			return 1
		}
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		fmt.Fprintln(stderr, "plots:", err)
		return 1
	}
	for name, c := range p.files {
		if err := os.WriteFile(filepath.Join(out, name), []byte(c), 0o644); err != nil {
			fmt.Fprintln(stderr, "plots:", err)
			return 1
		}
	}
	if err := os.WriteFile(docPath, []byte(doc), 0o644); err != nil {
		fmt.Fprintln(stderr, "plots:", err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote %d charts, %d skipped -> %s\n", len(p.charts), len(p.skipped), docPath)
	for _, s := range p.skipped {
		fmt.Fprintf(stdout, "  skipped: %s: %s\n", s[0], s[1])
	}
	if len(p.skipped) > 0 {
		return 1
	}
	return 0
}

// ---------------------------------------------------------------- per-folder summaries
//
// Each was a small Python script next to the data it summarised; each port
// prints the committed table again from the committed inputs.

type summaryCmd struct {
	usage string
	run   func(args []string, w *strings.Builder) (problems int, err error)
}

var summaries = map[string]summaryCmd{
	"bench-ops":    {"bench-ops [-parity FILE] LOG...", benchOps},
	"cpu-ab":       {"cpu-ab OUT_DIR", cpuAB},
	"cpu-ab-linux": {"cpu-ab-linux OUT_DIR", cpuABLinux},
	"attn-ab":      {"attn-ab AB_TXT", attnAB},
	"bpe":          {"bpe OUT_DIR BASE_LABEL", bpeSummary},
	"decode-ab":    {"decode-ab OUT_DIR [KEY]", decodeAB},
	"sweep-fit":    {"sweep-fit OUT_DIR", sweepFit},
}

var errUsage = errors.New("usage")

func runSummary(s summaryCmd, args []string, stdout, stderr io.Writer) int {
	var w strings.Builder
	problems, err := s.run(args, &w)
	if errors.Is(err, errUsage) {
		fmt.Fprintln(stderr, "usage: benchreport", s.usage)
		return 2
	}
	if err != nil {
		fmt.Fprintln(stderr, "benchreport:", err)
		return 1
	}
	io.WriteString(stdout, w.String())
	if problems > 0 {
		fmt.Fprintf(stderr, "benchreport: %d record(s) were neither ok nor info; see above\n", problems)
		return 1
	}
	return 0
}

func pl(w *strings.Builder, s string) { w.WriteString(s); w.WriteByte('\n') }

// pyPct is Python's format(x, ".0%").
func pyPct(x float64) string { return ff(x*100, 0) + "%" }

// pyE is Python's format(x, ".<prec>e"), which spells nan and inf in lower case.
func pyE(x float64, prec int) string {
	switch {
	case math.IsNaN(x):
		return "nan"
	case math.IsInf(x, 1):
		return "inf"
	case math.IsInf(x, -1):
		return "-inf"
	}
	return strconv.FormatFloat(x, 'e', prec, 64)
}

// pyQuote is Python's repr() of a str.
func pyQuote(s string) string {
	q := byte('\'')
	if strings.ContainsRune(s, '\'') && !strings.ContainsRune(s, '"') {
		q = '"'
	}
	var b strings.Builder
	b.WriteByte(q)
	for _, r := range s {
		switch {
		case r == '\\':
			b.WriteString(`\\`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r == rune(q):
			b.WriteByte('\\')
			b.WriteByte(q)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(&b, `\x%02x`, r)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte(q)
	return b.String()
}

// pySum is Python 3.12+'s sum() over ints and floats, starting from int 0:
// ints add exactly until the first float, then floats use Neumaier's
// compensated sum and ints are added plainly, and the compensation is added
// at the end.
type pyNumber struct {
	f     float64
	isInt bool
	i     int64
}

func pySum(items []pyNumber) float64 {
	var iResult int64
	i := 0
	for ; i < len(items) && items[i].isInt; i++ {
		iResult += items[i].i
	}
	if i == len(items) {
		return float64(iResult)
	}
	fResult := float64(iResult) + items[i].f
	c := 0.0
	for _, it := range items[i+1:] {
		if it.isInt {
			fResult += float64(it.i)
			continue
		}
		x := it.f
		t := fResult + x
		if math.Abs(fResult) >= math.Abs(x) {
			c += (fResult - t) + x
		} else {
			c += (x - t) + fResult
		}
		fResult = t
	}
	if c != 0 && !math.IsInf(c, 0) && !math.IsNaN(c) {
		fResult += c
	}
	return fResult
}

// splitKV is `dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)`.
func splitKV(line string) map[string]string {
	out := map[string]string{}
	fs := strings.FieldsFunc(line, pyIsSpace)
	for _, tok := range fs[min(1, len(fs)):] {
		if k, v, ok := strings.Cut(tok, "="); ok {
			out[k] = v
		}
	}
	return out
}

func need(m map[string]string, k string) (string, error) {
	v, ok := m[k]
	if !ok {
		return "", fmt.Errorf("KeyError: %q", k)
	}
	return v, nil
}

func needF(m map[string]string, k string) (float64, error) {
	v, err := need(m, k)
	if err != nil {
		return 0, err
	}
	return pyFloat(v)
}

// ------------------------------------------- bench-ops: ojas-cpu bench_ops.rs / torch_ops.py rounds
//
// Each round log holds the OJAS_OP lines of one bench_ops run or the TORCH_OP
// lines of one torch_ops.py run. The parity log holds PARITY and PARITY_F64
// lines from `torch_ops.py --parity`. Per row the statistic is the minimum
// over rounds of each round's min-of-N; spread is (max round min - min round
// min) / min round min, flagged above 15%. Ratio is torch / ojas, so below 1
// means ojas is slower. A row with no parity line prints MISSING.

var (
	opsPrimary = map[string]string{"sdpa": "math", "block": "math"}
	opsAlt     = map[string]string{"sdpa": "default", "block": "default", "adamw_768x768": "fused",
		"adamw_3072x768": "fused", "adamw_50304x768": "fused"}
)

const opsSpreadFlag = 0.15

type opCount struct {
	op, dir string
	n       int64
}

// How many times each op row runs in one block step as bench_ops.rs composes
// it (forward plus the hand-written backward). Residual-gradient sums in the
// backward use add.fwd.
var (
	blockFwd = []opCount{{"rms_norm", "fwd", 2}, {"linear_qkvo", "fwd", 4}, {"linear_up", "fwd", 2},
		{"linear_down", "fwd", 1}, {"qk_norm", "fwd", 1}, {"rope", "fwd", 2},
		{"vres", "fwd", 1}, {"permute", "fwd", 4}, {"sdpa", "fwd", 1}, {"gate", "fwd", 1},
		{"add", "fwd", 2}, {"silu", "fwd", 1}, {"mul", "fwd", 1}}
	blockBwd = []opCount{{"rms_norm", "bwd", 2}, {"linear_qkvo", "bwd", 4}, {"linear_up", "bwd", 2},
		{"linear_down", "bwd", 1}, {"qk_norm", "bwd", 1}, {"rope", "bwd", 2},
		{"vres", "bwd", 1}, {"permute", "fwd", 4}, {"sdpa", "bwd", 1}, {"gate", "bwd", 1},
		{"add", "fwd", 6}, {"silu", "bwd", 1}, {"mul", "bwd", 1}}
)

type parityLine struct {
	out string
	ok  bool
	err float64
	tol string
}

func opsFmt(ms *float64) string {
	if ms == nil {
		return "—"
	}
	if *ms < 10 {
		return ff(*ms, 3)
	}
	return ff(*ms, 1)
}

// opsBest is (min, (max - min) / min) of the round minimums.
func opsBest(series []float64) (*float64, *float64) {
	if len(series) == 0 {
		return nil, nil
	}
	lo, hi := minOf(series), maxOf(series)
	s := 0.0
	if lo > 0 {
		s = (hi - lo) / lo
	}
	return &lo, &s
}

// pyMaxF is Python's max() over floats: the first maximal element, so a NaN
// seen first stays.
func pyMaxF(xs []float64) float64 {
	m := xs[0]
	for _, x := range xs[1:] {
		if x > m {
			m = x
		}
	}
	return m
}

func benchOps(args []string, w *strings.Builder) (int, error) {
	fs := flag.NewFlagSet("bench-ops", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	parityPath := fs.String("parity", "", "parity log from torch_ops.py --parity")
	if err := fs.Parse(args); err != nil || fs.NArg() == 0 {
		return 0, errUsage
	}
	type otKey struct {
		op, dir string
		t       int
	}
	type tvKey struct{ op, dir, variant string }
	type odKey struct{ op, dir string }
	ojas := map[otKey][]float64{}
	torch := map[tvKey][]float64{}
	shapes := map[odKey]string{}
	var order []odKey
	seen := map[odKey]bool{}
	threadSet := map[int]bool{}
	for _, path := range fs.Args() {
		lines, err := fileLines(path)
		if err != nil {
			return 0, err
		}
		for _, line := range lines {
			switch {
			case strings.HasPrefix(line, "OJAS_OP "):
				d := splitKV(line)
				var v [4]string
				for i, k := range []string{"op", "dir", "shape", "threads"} {
					if v[i], err = need(d, k); err != nil {
						return 0, fmt.Errorf("%s: %v", path, err)
					}
				}
				t, err := strconv.Atoi(v[3])
				if err != nil {
					return 0, fmt.Errorf("%s: threads: %v", path, err)
				}
				ms, err := needF(d, "min_ms")
				if err != nil {
					return 0, fmt.Errorf("%s: %v", path, err)
				}
				k := odKey{v[0], v[1]}
				if !seen[k] {
					seen[k] = true
					order = append(order, k)
				}
				shapes[k] = v[2]
				threadSet[t] = true
				ojas[otKey{v[0], v[1], t}] = append(ojas[otKey{v[0], v[1], t}], ms)
			case strings.HasPrefix(line, "TORCH_OP "):
				d := splitKV(line)
				var v [3]string
				for i, k := range []string{"op", "dir", "variant"} {
					if v[i], err = need(d, k); err != nil {
						return 0, fmt.Errorf("%s: %v", path, err)
					}
				}
				ms, err := needF(d, "min_ms")
				if err != nil {
					return 0, fmt.Errorf("%s: %v", path, err)
				}
				torch[tvKey{v[0], v[1], v[2]}] = append(torch[tvKey{v[0], v[1], v[2]}], ms)
			}
		}
	}
	prow := map[tvKey][]parityLine{}
	type f64Key struct{ op, dir, out string }
	var f64Order []f64Key
	f64 := map[f64Key]map[string]string{}
	if *parityPath != "" {
		lines, err := fileLines(*parityPath)
		if err != nil {
			return 0, err
		}
		for _, line := range lines {
			switch {
			case strings.HasPrefix(line, "PARITY "):
				d := splitKV(line)
				ne, ok := d["norm_err"]
				if !ok {
					ne = "nan"
				}
				e, err := pyFloat(ne)
				if err != nil {
					return 0, err
				}
				var v [6]string
				for i, k := range []string{"op", "dir", "variant", "out", "ok", "tol"} {
					if v[i], err = need(d, k); err != nil {
						return 0, fmt.Errorf("%s: %v", *parityPath, err)
					}
				}
				k := tvKey{v[0], v[1], v[2]}
				prow[k] = append(prow[k], parityLine{v[3], v[4] == "True", e, v[5]})
			case strings.HasPrefix(line, "PARITY_F64 "):
				d := splitKV(line)
				var v [3]string
				for i, k := range []string{"op", "dir", "out"} {
					if v[i], err = need(d, k); err != nil {
						return 0, fmt.Errorf("%s: %v", *parityPath, err)
					}
				}
				k := f64Key{v[0], v[1], v[2]}
				if _, ok := f64[k]; !ok {
					f64Order = append(f64Order, k)
				}
				f64[k] = d
			}
		}
	}
	parityCell := func(op, d, variant string) (string, error) {
		lines := prow[tvKey{op, d, variant}]
		if len(lines) == 0 {
			return "MISSING", nil
		}
		var errs []float64
		for _, l := range lines {
			errs = append(errs, l.err)
		}
		worst, tol := pyMaxF(errs), lines[0].tol
		var bad []parityLine
		for _, l := range lines {
			if !l.ok {
				bad = append(bad, l)
			}
		}
		if len(bad) == 0 {
			return "pass " + pyE(worst, 1) + " (tol " + tol + ")", nil
		}
		// Torch errors an f64 arbiter already pinned on torch: an output scaled by
		// that torch value (clip's gradients) inherits the same relative error.
		var torchOff []float64
		for _, k := range f64Order {
			a := f64[k]
			if k.op != op || k.dir != d {
				continue
			}
			ok, err := need(a, "ojas_ok")
			if err != nil {
				return "", err
			}
			if ok == "True" {
				te, err := needF(a, "torch_err")
				if err != nil {
					return "", err
				}
				torchOff = append(torchOff, te)
			}
		}
		var notes []string
		for _, b := range bad {
			arb, has := f64[f64Key{op, d, b.out}]
			inherits := false
			for _, te := range torchOff {
				if math.Abs(b.err-te) <= 0.01*te {
					inherits = true
					break
				}
			}
			arbOK := false
			if has {
				s, err := need(arb, "ojas_ok")
				if err != nil {
					return "", err
				}
				arbOK = s == "True"
			}
			switch {
			case !has && inherits:
				notes = append(notes, b.out+": "+pyE(b.err, 1)+", inherits torch's f64 miss above")
			case has && arbOK:
				te, err := needF(arb, "torch_err")
				if err != nil {
					return "", err
				}
				oe, err := needF(arb, "ojas_err")
				if err != nil {
					return "", err
				}
				notes = append(notes, b.out+": torch off f64 by "+pyE(te, 1)+", ojas "+pyE(oe, 1))
			case has:
				oe, err := needF(arb, "ojas_err")
				if err != nil {
					return "", err
				}
				te, err := needF(arb, "torch_err")
				if err != nil {
					return "", err
				}
				notes = append(notes, "FAIL "+b.out+": ojas off f64 by "+pyE(oe, 1)+" (torch "+pyE(te, 1)+")")
			default:
				notes = append(notes, "FAIL "+b.out+" "+pyE(b.err, 1))
			}
		}
		status := "pass vs f64"
		for _, n := range notes {
			if strings.HasPrefix(n, "FAIL") {
				status = "FAIL"
				break
			}
		}
		return status + " (tol " + tol + "); " + strings.Join(notes, "; "), nil
	}

	var threads []int
	for t := range threadSet {
		threads = append(threads, t)
	}
	sort.Ints(threads)
	oRounds, tRounds := 0, 0
	for _, v := range ojas {
		oRounds = max(oRounds, len(v))
	}
	for _, v := range torch {
		tRounds = max(tRounds, len(v))
	}
	pl(w, fmt.Sprintf("rounds: ojas %d, torch %d; times in ms, min over rounds", oRounds, tRounds))
	head := []string{"op", "dir", "shape"}
	for _, t := range threads {
		head = append(head, fmt.Sprintf("ojas %dt", t))
	}
	head = append(head, "torch")
	for _, t := range threads {
		head = append(head, fmt.Sprintf("ratio %dt", t))
	}
	head = append(head, "torch alt", "spread>15%", "parity")
	pl(w, "| "+strings.Join(head, " | ")+" |")
	pl(w, "|"+strings.Repeat("---|", len(head)))
	mins := map[string]*float64{} // "op\x00dir\x00t" -> min, t a thread count or "torch"
	mk := func(op, d, t string) string { return op + "\x00" + d + "\x00" + t }
	for _, k := range order {
		op, d := k.op, k.dir
		variant, ok := opsPrimary[op]
		if !ok {
			variant = "default"
		}
		var flags, cells []string
		for _, t := range threads {
			m, s := opsBest(ojas[otKey{op, d, t}])
			mins[mk(op, d, strconv.Itoa(t))] = m
			cells = append(cells, opsFmt(m))
			if s != nil && *s > opsSpreadFlag {
				flags = append(flags, fmt.Sprintf("ojas%dt %s", t, pyPct(*s)))
			}
		}
		tm, ts := opsBest(torch[tvKey{op, d, variant}])
		mins[mk(op, d, "torch")] = tm
		if ts != nil && *ts > opsSpreadFlag {
			flags = append(flags, "torch "+pyPct(*ts))
		}
		var ratios []string
		for _, t := range threads {
			om := mins[mk(op, d, strconv.Itoa(t))]
			if tm != nil && *tm != 0 && om != nil && *om != 0 {
				ratios = append(ratios, ff(*tm / *om, 2))
			} else {
				ratios = append(ratios, "—")
			}
		}
		altCell := "—"
		if alt, ok := opsAlt[op]; ok {
			if am, _ := opsBest(torch[tvKey{op, d, alt}]); am != nil {
				altCell = opsFmt(am) + " (" + alt + ")"
			}
		}
		tcell := opsFmt(tm)
		if variant != "default" {
			tcell += " (" + variant + ")"
		}
		flag := strings.Join(flags, ", ")
		if flag == "" {
			flag = "no"
		}
		pc, err := parityCell(op, d, variant)
		if err != nil {
			return 0, err
		}
		row := append([]string{op, d, shapes[k]}, cells...)
		row = append(append(append(row, tcell), ratios...), altCell, flag, pc)
		pl(w, "| "+strings.Join(row, " | ")+" |")
	}
	ts := []string{}
	for _, t := range threads {
		ts = append(ts, strconv.Itoa(t))
	}
	for _, t := range append(ts, "torch") {
		total := func(counts []opCount) float64 {
			var items []pyNumber
			for _, c := range counts {
				m := mins[mk(c.op, c.dir, t)]
				if m == nil || *m == 0 { // `or 0`: an int zero
					items = append(items, pyNumber{isInt: true})
				} else {
					items = append(items, pyNumber{f: float64(float64(c.n) * *m)})
				}
			}
			return pySum(items)
		}
		f, b := total(blockFwd), total(blockBwd)
		fwd, step := mins[mk("block", "fwd", t)], mins[mk("block", "step", t)]
		label := "torch"
		if t != "torch" {
			label = "ojas " + t + "t"
		}
		diff := "—"
		if fwd != nil && *fwd != 0 && step != nil && *step != 0 {
			diff = opsFmt(fp(*step - *fwd))
		}
		pl(w, fmt.Sprintf("block, %s: sum of op rows fwd %s + bwd %s = %s ms; measured fwd %s, step %s, step - fwd %s",
			label, ff(f, 2), ff(b, 2), ff(f+b, 2), opsFmt(fwd), opsFmt(step), diff))
	}
	return 0, nil
}

// ------------------------------------------- cpu-ab: 2026-10-08-cpu-hot-paths run.sh output
//
// Per row, the minimum over rounds of each lane's per-round minimum, the
// after/base ratio by that minimum and by the median of the per-round ratios,
// and after/torch where torch has the row.

var cpuABFile = regexp.MustCompile(`^(ops|tape)-(\d+)-(\w+)\.txt`)

func cpuAB(args []string, w *strings.Builder) (int, error) {
	if len(args) != 1 {
		return 0, errUsage
	}
	type rk struct{ a, b string }
	// lanes[lane][round][key] = min_ms
	lanes := map[string]map[int]map[rk]float64{}
	for _, path := range globSorted(filepath.Join(args[0], "*-*-*.txt")) {
		m := cpuABFile.FindStringSubmatch(filepath.Base(path))
		if m == nil {
			continue
		}
		kind, lane := m[1], m[3]
		rnd, err := strconv.Atoi(m[2])
		if err != nil {
			return 0, err
		}
		prefix := "OJAS_TAPE"
		if kind == "ops" {
			prefix = "OJAS_OP"
			if lane == "torch" {
				prefix = "TORCH_OP"
			}
		}
		lines, err := fileLines(path)
		if err != nil {
			return 0, err
		}
		if lanes[lane] == nil {
			lanes[lane] = map[int]map[rk]float64{}
		}
		if lanes[lane][rnd] == nil {
			lanes[lane][rnd] = map[rk]float64{}
		}
		for _, line := range lines {
			if !strings.HasPrefix(line, prefix) {
				continue
			}
			f := splitKV(line)
			var key rk
			if prefix == "OJAS_TAPE" {
				row, err := need(f, "row")
				if err != nil {
					return 0, fmt.Errorf("%s: %v", path, err)
				}
				key = rk{"tape", row}
			} else {
				op, err := need(f, "op")
				if err != nil {
					return 0, fmt.Errorf("%s: %v", path, err)
				}
				dir, err := need(f, "dir")
				if err != nil {
					return 0, fmt.Errorf("%s: %v", path, err)
				}
				if v, ok := f["variant"]; ok {
					dir += "/" + v
				}
				key = rk{op, dir}
			}
			ms, err := needF(f, "min_ms")
			if err != nil {
				return 0, fmt.Errorf("%s: %v", path, err)
			}
			lanes[lane][rnd][key] = ms
		}
	}
	sortedRounds := func(m map[int]map[rk]float64) []int {
		var rs []int
		for r := range m {
			rs = append(rs, r)
		}
		sort.Ints(rs)
		return rs
	}
	rounds := sortedRounds(lanes["after"])
	keySet := map[rk]bool{}
	for _, r := range lanes["after"] {
		for k := range r {
			keySet[k] = true
		}
	}
	var keys []rk
	for k := range keySet {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].a != keys[j].a {
			return keys[i].a < keys[j].a
		}
		return keys[i].b < keys[j].b
	})
	pl(w, fmt.Sprintf("rounds=%d; cells are ms, min over rounds of each round's min-of-N\n", len(rounds)))
	pl(w, "| row | dir | base | after | after/base (min) | after/base (median of rounds) | torch | after/torch |")
	pl(w, "| :-- | :-- | --: | --: | --: | --: | --: | --: |")
	// series is [lanes[lane][r].get(key) for r in rounds if r in lanes[lane]].
	series := func(lane string, key rk) []*float64 {
		var out []*float64
		for _, r := range rounds {
			rd, ok := lanes[lane][r]
			if !ok {
				continue
			}
			if v, ok := rd[key]; ok {
				out = append(out, fp(v))
			} else {
				out = append(out, nil)
			}
		}
		return out
	}
	minNN := func(xs []*float64) *float64 {
		var m *float64
		for _, x := range xs {
			if x != nil && (m == nil || *x < *m) {
				m = x
			}
		}
		return m
	}
	f4 := func(v *float64) string {
		if v == nil {
			return ""
		}
		return ff(*v, 4)
	}
	ratio := func(x, y *float64) string {
		if x == nil || y == nil {
			return ""
		}
		return ff(*x / *y, 2)
	}
	for _, key := range keys {
		b, a := series("base", key), series("after", key)
		var per []float64
		for i := 0; i < len(a) && i < len(b); i++ {
			if a[i] != nil && b[i] != nil {
				per = append(per, *a[i] / *b[i])
			}
		}
		bmin, amin := minNN(b), minNN(a)
		tkey := key
		if key.a == "adamw_50304x768" {
			tkey = rk{key.a, key.b + "/fused"}
		}
		t := series("torch", tkey)
		allNil := true
		for _, v := range t {
			if v != nil {
				allNil = false
			}
		}
		if allNil {
			t = series("torch", rk{key.a, key.b + "/default"})
		}
		tmin := minNN(t)
		med := ""
		if len(per) > 0 {
			med = ff(median(per), 2)
		}
		pl(w, fmt.Sprintf("| %s | %s | %s | %s | %s | %s | %s | %s |", key.a, key.b, f4(bmin), f4(amin), ratio(amin, bmin), med, f4(tmin), ratio(amin, tmin)))
	}
	return 0, nil
}

// ------------------------------------------- cpu-ab-linux: 2026-10-10-cpu-linux-hot-paths run.sh output

func cpuABLinux(args []string, w *strings.Builder) (int, error) {
	if len(args) != 1 {
		return 0, errUsage
	}
	type key struct {
		op, dir string
		t       int
	}
	best := map[key]map[string]float64{}
	for _, path := range globSorted(filepath.Join(args[0], "ops-*-*.txt")) {
		stem := strings.TrimSuffix(filepath.Base(path), ".txt")
		lane := stem[strings.LastIndexByte(stem, '-')+1:]
		lines, err := fileLines(path)
		if err != nil {
			return 0, err
		}
		for _, line := range lines {
			if !strings.HasPrefix(line, "OJAS_OP ") {
				continue
			}
			f := kvOf(kvRe, line)
			var v [3]string
			for i, k := range []string{"op", "dir", "threads"} {
				if v[i], err = need(f, k); err != nil {
					return 0, fmt.Errorf("%s: %v", path, err)
				}
			}
			t, err := strconv.Atoi(v[2])
			if err != nil {
				return 0, fmt.Errorf("%s: threads: %v", path, err)
			}
			ms, err := needF(f, "min_ms")
			if err != nil {
				return 0, fmt.Errorf("%s: %v", path, err)
			}
			k := key{v[0], v[1], t}
			if best[k] == nil {
				best[k] = map[string]float64{}
			}
			if old, ok := best[k][lane]; !ok || ms < old {
				best[k][lane] = ms
			}
		}
	}
	var keys []key
	for k := range best {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := keys[i], keys[j]
		if a.op != b.op {
			return a.op < b.op
		}
		if a.dir != b.dir {
			return a.dir < b.dir
		}
		return a.t < b.t
	})
	pl(w, "| op | dir | threads | base ms | after ms | after/base |")
	pl(w, "| :-- | :-- | --: | --: | --: | --: |")
	for _, k := range keys {
		b, okB := best[k]["base"]
		a, okA := best[k]["after"]
		if !okB || !okA {
			return 0, fmt.Errorf("TypeError: %s %s %d lacks a base or after cell", k.op, k.dir, k.t)
		}
		ratio := "-"
		if a != 0 && b != 0 {
			ratio = ff(a/b, 2)
		}
		pl(w, fmt.Sprintf("| %s | %s | %d | %s | %s | %s |", k.op, k.dir, k.t, ff(b, 3), ff(a, 3), ratio))
	}
	return 0, nil
}

// ------------------------------------------- attn-ab: per-round new/old ratios of the attention A/B

func attnAB(args []string, w *strings.Builder) (int, error) {
	if len(args) != 1 {
		return 0, errUsage
	}
	lines, err := fileLines(args[0])
	if err != nil {
		return 0, err
	}
	type key struct{ be, shape, w string }
	type rt struct {
		r    int
		tree string
	}
	rows := map[key]map[rt]map[string]float64{}
	for _, line := range lines {
		if !strings.HasPrefix(line, "AB ") {
			continue
		}
		f := kvOf(kvRe, line)
		var v [5]string
		for i, k := range []string{"backend", "shape", "window", "round", "tree"} {
			if v[i], err = need(f, k); err != nil {
				return 0, err
			}
		}
		r, err := strconv.Atoi(v[3])
		if err != nil {
			return 0, err
		}
		fields := map[string]float64{}
		for k, s := range f {
			if p := prefix(k, 3); p == "fwd" || p == "bwd" || p == "ctl" {
				if fields[k], err = pyFloat(s); err != nil {
					return 0, err
				}
			}
		}
		k := key{v[0], v[1], v[2]}
		if rows[k] == nil {
			rows[k] = map[rt]map[string]float64{}
		}
		rows[k][rt{r, v[4]}] = fields
	}
	var keys []key
	for k := range rows {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := keys[i], keys[j]
		if a.be != b.be {
			return a.be < b.be
		}
		if a.shape != b.shape {
			return a.shape < b.shape
		}
		return a.w < b.w
	})
	field := func(m map[string]float64, k string) (float64, error) {
		v, ok := m[k]
		if !ok {
			return 0, fmt.Errorf("KeyError: %q", k)
		}
		return v, nil
	}
	pl(w, "| backend | shape | metric | old min ms (best) | new min ms (best) | new/old per round (median, range) | control new/old (median, range) |")
	pl(w, "|:--|:--|:--|--:|--:|:--|:--|")
	for _, k := range keys {
		if k.w != "full" {
			continue
		}
		d := rows[k]
		rset := map[int]bool{}
		for x := range d {
			rset[x.r] = true
		}
		var rounds []int
		for r := range rset {
			rounds = append(rounds, r)
		}
		sort.Ints(rounds)
		for _, metric := range []string{"bwd_min", "fwd_min"} {
			var rat, ctl, olds, news []float64
			for _, r := range rounds {
				o, okO := d[rt{r, "old"}]
				n, okN := d[rt{r, "new"}]
				if okO {
					v, err := field(o, metric)
					if err != nil {
						return 0, err
					}
					olds = append(olds, v)
				}
				if okN {
					v, err := field(n, metric)
					if err != nil {
						return 0, err
					}
					news = append(news, v)
				}
				if okO && okN {
					var v [4]float64
					for i, x := range []struct {
						m map[string]float64
						k string
					}{{n, metric}, {o, metric}, {n, "ctl_min"}, {o, "ctl_min"}} {
						if v[i], err = field(x.m, x.k); err != nil {
							return 0, err
						}
					}
					rat, ctl = append(rat, v[0]/v[1]), append(ctl, v[2]/v[3])
				}
			}
			if len(olds) == 0 || len(news) == 0 || len(rat) == 0 {
				return 0, fmt.Errorf("%s %s: no paired rounds", k.be, k.shape)
			}
			pl(w, fmt.Sprintf("| %s | %s | %s | %s | %s | %s (%s-%s) | %s (%s-%s) |", k.be, k.shape, metric[:3],
				ff(minOf(olds), 3), ff(minOf(news), 3),
				ff(median(rat), 3), ff(minOf(rat), 3), ff(maxOf(rat), 3),
				ff(median(ctl), 3), ff(minOf(ctl), 3), ff(maxOf(ctl), 3)))
		}
	}
	pl(w, "")
	pl(w, "| backend | shape | window | bwd min ms (best of rounds) | fwd min ms (best) |")
	pl(w, "|:--|:--|:--|--:|--:|")
	for _, k := range keys {
		var bw, fw []float64
		for x, v := range rows[k] {
			if x.tree != "new" {
				continue
			}
			b, err := field(v, "bwd_min")
			if err != nil {
				return 0, err
			}
			f, err := field(v, "fwd_min")
			if err != nil {
				return 0, err
			}
			bw, fw = append(bw, b), append(fw, f)
		}
		if len(bw) == 0 {
			continue
		}
		pl(w, fmt.Sprintf("| %s | %s | %s | %s | %s |", k.be, k.shape, k.w, ff(minOf(bw), 3), ff(minOf(fw), 3)))
	}
	return 0, nil
}

// ------------------------------------------- bpe: bench/bpe_throughput.rs run.sh output
//
// Per lane and row: the minimum over rounds of each process's min, the MB/s
// and Mtok/s at that minimum, and the median over rounds of (lane min / base
// min) within the same round. Digests must agree across lanes.

var (
	bpeFile   = regexp.MustCompile(`^r(\d+)_(.+)\.txt`)
	bpeDigest = regexp.MustCompile(`digest=(\w+)`)
	bpeParity = regexp.MustCompile(`tiktoken_parity=(\S+) documents_matched=(\S+)`)
)

func bpeSummary(args []string, w *strings.Builder) (int, error) {
	if len(args) != 2 {
		return 0, errUsage
	}
	base := args[1]
	type cellv struct{ s, mbs, mtok float64 }
	type run struct {
		order  []string
		rows   map[string]cellv
		digest string
		parity [2]string
	}
	var laneOrder []string
	runs := map[string]map[int]*run{}
	roundOrder := map[string][]int{}
	for _, path := range globSorted(filepath.Join(args[0], "r*_*.txt")) {
		m := bpeFile.FindStringSubmatch(filepath.Base(path))
		if m == nil {
			return 0, fmt.Errorf("AttributeError: %s is not r<N>_<lane>.txt", path)
		}
		rnd, _ := strconv.Atoi(m[1])
		label := m[2]
		text, err := readText(path)
		if err != nil {
			return 0, err
		}
		rn := &run{rows: map[string]cellv{}}
		for _, line := range strings.Split(text, "\n") {
			cells := strings.Split(strings.Trim(line, "|"), "|")
			for i := range cells {
				cells[i] = pyStrip(cells[i])
			}
			if len(cells) == 5 && (strings.HasPrefix(cells[0], "encode") || strings.HasPrefix(cells[0], "decode")) {
				v, err := floats(cells[1], cells[3], cells[4])
				if err != nil {
					return 0, fmt.Errorf("%s: %v", path, err)
				}
				if _, ok := rn.rows[cells[0]]; !ok {
					rn.order = append(rn.order, cells[0])
				}
				rn.rows[cells[0]] = cellv{v[0], v[1], v[2]}
			}
		}
		dg := bpeDigest.FindStringSubmatch(text)
		pa := bpeParity.FindStringSubmatch(text)
		if dg == nil || pa == nil {
			return 0, fmt.Errorf("%s: no digest= or tiktoken_parity= line", path)
		}
		rn.digest, rn.parity = dg[1], [2]string{pa[1], pa[2]}
		if runs[label] == nil {
			runs[label] = map[int]*run{}
			laneOrder = append(laneOrder, label)
		}
		if _, ok := runs[label][rnd]; !ok {
			roundOrder[label] = append(roundOrder[label], rnd)
		}
		runs[label][rnd] = rn
	}
	digests, parities := map[string]bool{}, map[[2]string]bool{}
	for _, l := range laneOrder {
		for _, r := range runs[l] {
			digests[r.digest] = true
			parities[r.parity] = true
		}
	}
	var ps [][2]string
	for p := range parities {
		ps = append(ps, p)
	}
	sort.Slice(ps, func(i, j int) bool {
		if ps[i][0] != ps[j][0] {
			return ps[i][0] < ps[j][0]
		}
		return ps[i][1] < ps[j][1]
	})
	var pss []string
	for _, p := range ps {
		pss = append(pss, "("+pyQuote(p[0])+", "+pyQuote(p[1])+")")
	}
	// sorted() sorts the raw strings; the list repr then quotes each.
	var raw, ds []string
	for d := range digests {
		raw = append(raw, d)
	}
	sort.Strings(raw)
	for _, d := range raw {
		ds = append(ds, pyQuote(d))
	}
	pl(w, "digests: ["+strings.Join(ds, ", ")+"]  parity: ["+strings.Join(pss, ", ")+"]")
	baseRuns, ok := runs[base]
	if !ok {
		return 0, fmt.Errorf("KeyError: no lane %q", base)
	}
	rounds := append([]int(nil), roundOrder[base]...)
	sort.Ints(rounds)
	rowNames := baseRuns[rounds[0]].order
	pl(w, "")
	pl(w, "| lane | row | min s | MB/s | Mtok/s | lane/base (min of mins) | median of round ratios |")
	pl(w, "| :-- | :-- | --: | --: | --: | --: | --: |")
	for _, label := range laneOrder {
		lane := runs[label]
		for _, row := range rowNames {
			var best *cellv
			var ratios []float64
			for _, r := range roundOrder[label] {
				c, ok := lane[r].rows[row]
				if !ok {
					return 0, fmt.Errorf("KeyError: lane %s round %d lacks %s", label, r, row)
				}
				if best == nil || c.s < best.s {
					cc := c
					best = &cc
				}
				if br, ok := baseRuns[r]; ok {
					bc, ok := br.rows[row]
					if !ok {
						return 0, fmt.Errorf("KeyError: base round %d lacks %s", r, row)
					}
					ratios = append(ratios, c.s/bc.s)
				}
			}
			var baseMins []float64
			for _, r := range roundOrder[base] {
				bc, ok := baseRuns[r].rows[row]
				if !ok {
					return 0, fmt.Errorf("KeyError: base round %d lacks %s", r, row)
				}
				baseMins = append(baseMins, bc.s)
			}
			if len(ratios) == 0 {
				return 0, fmt.Errorf("lane %s shares no round with %s", label, base)
			}
			pl(w, fmt.Sprintf("| %s | %s | %s | %s | %s | %s | %s |", label, row, ff(best.s, 4), ff(best.mbs, 2), ff(best.mtok, 3),
				ff(best.s/minOf(baseMins), 3), ff(median(ratios), 3)))
		}
	}
	return 0, nil
}

// ------------------------------------------- decode-ab: ab_decode.sh before/after JSONL
//
// Per lane and row: the median over rounds of each side's KEY, after/before
// per round (median [min-max]), and decode ms per token = (decode row -
// prefill row) / 31 per side.

func decodeAB(args []string, w *strings.Builder) (int, error) {
	if len(args) < 1 || len(args) > 2 {
		return 0, errUsage
	}
	key := "median_ms"
	if len(args) == 2 {
		key = args[1]
	}
	type lr struct{ lane, row string }
	rows := map[lr]map[int]map[string]float64{}
	problems := 0
	var probText []string
	for _, p := range globSorted(filepath.Join(args[0], "round*", "*.jsonl")) {
		r, err := roundOf(p)
		if err != nil {
			return 0, err
		}
		base := filepath.Base(p)
		parts := strings.Split(base[:max(len(base)-6, 0)], "-")
		if len(parts) != 2 {
			return 0, fmt.Errorf("ValueError: %s is not <lane>-<side>.jsonl", p)
		}
		lane, side := parts[0], parts[1]
		lines, err := fileLines(p)
		if err != nil {
			return 0, err
		}
		for i, line := range lines {
			d, err := decodeRecord(line)
			if err != nil {
				return 0, fmt.Errorf("%s:%d: %v", p, i+1, err)
			}
			st := d.get("status")
			if _, isStr := d["status"].(string); !isStr {
				st = "\x00" // neither ok nor info
			}
			if st == "ok" && d.has("median_ms") {
				row, err := d.str("row")
				if err != nil {
					return 0, err
				}
				v, err := d.num(key)
				if err != nil {
					return 0, fmt.Errorf("%s:%d: %v", p, i+1, err)
				}
				k := lr{lane, row}
				if rows[k] == nil {
					rows[k] = map[int]map[string]float64{}
				}
				if rows[k][r] == nil {
					rows[k][r] = map[string]float64{}
				}
				rows[k][r][side] = v
			} else if st != "info" && st != "ok" {
				problems++
				probText = append(probText, fmt.Sprintf("PROBLEM %s %s %d %s", lane, side, r, pyStrip(line)))
			}
		}
	}
	for _, t := range probText {
		pl(w, t)
	}
	var keys []lr
	for k := range rows {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].lane != keys[j].lane {
			return keys[i].lane < keys[j].lane
		}
		return keys[i].row < keys[j].row
	})
	pl(w, fmt.Sprintf("| lane | row | before %s | after %s | after / before per round, median [min-max] | rounds |", key, key))
	pl(w, "| :-- | :-- | --: | --: | :-- | --: |")
	for _, k := range keys {
		var bs, as, ratios []float64
		for _, x := range rows[k] {
			b, okB := x["before"]
			a, okA := x["after"]
			if okB && okA {
				bs, as, ratios = append(bs, b), append(as, a), append(ratios, a/b)
			}
		}
		if len(bs) == 0 {
			return 0, fmt.Errorf("StatisticsError: %s %s has no round with both sides", k.lane, k.row)
		}
		pl(w, fmt.Sprintf("| %s | %s | %s | %s | %s [%s-%s] | %d |", k.lane, k.row, ff(median(bs), 2), ff(median(as), 2),
			ff(median(ratios), 3), ff(minOf(ratios), 3), ff(maxOf(ratios), 3), len(bs)))
	}
	pl(w, "")
	pl(w, "| lane | decode row | before ms / token | after ms / token | after - before ms / token, per round median [min-max] |")
	pl(w, "| :-- | :-- | --: | --: | :-- |")
	plus := func(x float64) string { return fmt.Sprintf("%+.3f", x) }
	for _, lane := range []string{"metal", "wgpu"} {
		pre := rows[lr{lane, "gen_prefill_p32"}]
		for _, k := range keys {
			if k.lane != lane || k.row == "gen_prefill_p32" {
				continue
			}
			rs := rows[k]
			var rnds []int
			for r := range rs {
				rnds = append(rnds, r)
			}
			sort.Ints(rnds)
			tok := map[string][]float64{}
			var diffs []float64
			for _, r := range rnds {
				pr, ok := pre[r]
				if !ok {
					continue
				}
				t := map[string]float64{}
				for _, s := range []string{"before", "after"} {
					x, ok1 := rs[r][s]
					y, ok2 := pr[s]
					if ok1 && ok2 {
						t[s] = (x - y) / 31
						tok[s] = append(tok[s], t[s])
					}
				}
				if len(t) == 2 {
					diffs = append(diffs, t["after"]-t["before"])
				}
			}
			if len(tok["before"]) == 0 || len(tok["after"]) == 0 || len(diffs) == 0 {
				return 0, fmt.Errorf("StatisticsError: %s %s has no round paired with gen_prefill_p32", lane, k.row)
			}
			pl(w, fmt.Sprintf("| %s | %s | %s | %s | %s [%s-%s] |", lane, k.row, ff(median(tok["before"]), 3), ff(median(tok["after"]), 3),
				plus(median(diffs)), plus(minOf(diffs)), plus(maxOf(diffs))))
		}
	}
	return problems, nil
}

// ------------------------------------------- sweep-fit: the sweep_ rows of a run_paired.sh run
//
// Per runtime and row: m = median over rounds of the per-round minimum, d =
// median over rounds of the per-round median (the conventions of
// docs/bench-gpu-vs-torch.md). Spread = max / min of the per-round medians.
//
// silu: floor a = t(1); slope b from the two largest N; model crossover N* =
// a / b (fixed part = per-value part). The direct crossover is where the
// measured t(N) first reaches 2 * t(1), interpolated linearly between sweep
// points; it needs no model.
//
// decode: B requests batched (b) vs B batch-1 calls before one sync (x).

var sweepRuntimes = []string{"ojas-metal", "ojas-wgpu", "torch-mps"}

type sweepStat struct {
	status       string
	n            int
	m, d, spread float64
}

func sweepFit(args []string, w *strings.Builder) (int, error) {
	if len(args) != 1 {
		return 0, errUsage
	}
	type rr struct{ rt, row string }
	per := map[rr]map[int]record{}
	rowSet := map[string]bool{}
	for _, path := range globSorted(filepath.Join(args[0], "round*", "*.jsonl")) {
		r, err := roundOf(path)
		if err != nil {
			return 0, err
		}
		lines, err := fileLines(path)
		if err != nil {
			return 0, err
		}
		for i, line := range lines {
			line = pyStrip(line)
			if line == "" {
				continue
			}
			d, err := decodeRecord(line)
			if err != nil {
				return 0, fmt.Errorf("%s:%d: %v", path, i+1, err)
			}
			row, err := d.str("row")
			if err != nil {
				return 0, fmt.Errorf("%s:%d: %v", path, i+1, err)
			}
			if strings.HasPrefix(row, "_") {
				continue
			}
			rt, err := d.str("runtime")
			if err != nil {
				return 0, fmt.Errorf("%s:%d: %v", path, i+1, err)
			}
			if per[rr{rt, row}] == nil {
				per[rr{rt, row}] = map[int]record{}
			}
			per[rr{rt, row}][r] = d
			rowSet[row] = true
		}
	}
	stats := func(rt, row string) (sweepStat, error) {
		var bad []string
		badSeen := map[string]bool{}
		var mins, meds []float64
		for _, d := range per[rr{rt, row}] {
			st, err := d.str("status")
			if err != nil {
				return sweepStat{}, err
			}
			if st != "ok" {
				if !badSeen[st] {
					badSeen[st] = true
					bad = append(bad, st)
				}
				continue
			}
			mn, err := d.num("min_ms")
			if err != nil {
				return sweepStat{}, err
			}
			md, err := d.num("median_ms")
			if err != nil {
				return sweepStat{}, err
			}
			mins, meds = append(mins, mn), append(meds, md)
		}
		sort.Strings(bad)
		if len(bad) > 0 || len(mins) == 0 {
			s := strings.Join(bad, ",")
			if s == "" {
				s = "missing"
			}
			return sweepStat{status: s, n: len(mins)}, nil
		}
		return sweepStat{"ok", len(mins), median(mins), median(meds), maxOf(meds)/minOf(meds) - 1}, nil
	}
	f := func(x *float64, nd int) string {
		if x == nil {
			return "-"
		}
		return ff(*x, nd)
	}
	var rows []string
	for r := range rowSet {
		rows = append(rows, r)
	}
	sort.Strings(rows)
	type nrow struct {
		n   int
		row string
	}
	var silu []nrow
	bset := map[int]bool{}
	for _, r := range rows {
		if strings.HasPrefix(r, "sweep_silu_n") {
			parts := strings.Split(r, "_n")
			n, err := strconv.Atoi(parts[1])
			if err != nil {
				return 0, err
			}
			silu = append(silu, nrow{n, r})
		}
		if strings.HasPrefix(r, "sweep_decode_b") {
			parts := strings.Split(r, "_b")
			n, err := strconv.Atoi(parts[1])
			if err != nil {
				return 0, err
			}
			bset[n] = true
		}
	}
	sort.Slice(silu, func(i, j int) bool {
		if silu[i].n != silu[j].n {
			return silu[i].n < silu[j].n
		}
		return silu[i].row < silu[j].row
	})
	var bs []int
	for b := range bset {
		bs = append(bs, b)
	}
	sort.Ints(bs)

	pl(w, "## silu_forward over N values\n")
	pl(w, "Cells: min-of-round-minimums median (m) / median-of-medians (d) ms, spread of per-round medians.\n")
	pl(w, "| N | "+strings.Join(sweepRuntimes, " | ")+" | metal/torch d ratio |")
	pl(w, "| --: |"+strings.Repeat(" :-- |", len(sweepRuntimes))+" --: |")
	for _, nr := range silu {
		var cells []string
		s := map[string]sweepStat{}
		for _, rt := range sweepRuntimes {
			st, err := stats(rt, nr.row)
			if err != nil {
				return 0, err
			}
			s[rt] = st
			if st.status == "ok" {
				cells = append(cells, f(&st.m, 3)+" / "+f(&st.d, 3)+" ("+pyPct(st.spread)+")")
			} else {
				cells = append(cells, st.status)
			}
		}
		var ratio *float64
		if s["torch-mps"].status == "ok" && s["ojas-metal"].status == "ok" {
			ratio = fp(s["torch-mps"].d / s["ojas-metal"].d)
		}
		pl(w, fmt.Sprintf("| %d | ", nr.n)+strings.Join(cells, " | ")+" | "+f(ratio, 2)+"x |")
	}

	pl(w, "\n## Crossover where the per-value work equals the fixed cost\n")
	pl(w, "| runtime | stat | floor a = t(1) ms | slope b ns/value | model N* = a/b | direct N (t = 2a) |")
	pl(w, "| :-- | :-- | --: | --: | --: | --: |")
	for _, rt := range sweepRuntimes {
		for _, key := range []string{"m", "d"} {
			type p2 struct {
				n int
				t float64
			}
			var pts []p2
			for _, nr := range silu {
				st, err := stats(rt, nr.row)
				if err != nil {
					return 0, err
				}
				if st.status == "ok" {
					v := st.m
					if key == "d" {
						v = st.d
					}
					pts = append(pts, p2{nr.n, v})
				}
			}
			if len(pts) < 3 || pts[0].n != 1 {
				pl(w, fmt.Sprintf("| %s | %s | incomplete (%d sizes) | | | |", rt, key, len(pts)))
				continue
			}
			a := pts[0].t
			p0, p1 := pts[len(pts)-2], pts[len(pts)-1]
			b := (p1.t - p0.t) / float64(p1.n-p0.n)
			nstar := "-"
			if b > 0 {
				nstar = ff(a/b/1e6, 2) + "M"
			}
			dx := "-"
			for i := 0; i+1 < len(pts); i++ {
				n0, t0, n1, t1 := pts[i].n, pts[i].t, pts[i+1].n, pts[i+1].t
				if t1 >= 2*a && 2*a > t0 {
					x := float64(n0) + float64((2*a-t0)*float64(n1-n0))/(t1-t0)
					dx = ff(x/1e6, 2) + "M"
					break
				}
			}
			pl(w, fmt.Sprintf("| %s | %s | %s | %s | %s | %s |", rt, key, f(&a, 3), f(fp(b*1e6), 4), nstar, dx))
		}
	}

	pl(w, "\n## Decode: B requests batched (b) vs B batch-1 calls, one sync (x)\n")
	pl(w, "Cells: median-of-medians ms (spread). Per-request = cell / B. Extra per call = (x - b) / (B - 1).\n")
	pl(w, "| B | runtime | batched b | per request | split x | per request | extra per call |")
	pl(w, "| --: | :-- | --: | --: | --: | --: | --: |")
	for _, b := range bs {
		for _, rt := range sweepRuntimes {
			sb, err := stats(rt, fmt.Sprintf("sweep_decode_b%d", b))
			if err != nil {
				return 0, err
			}
			cb, pb := sb.status, "-"
			if sb.status == "ok" {
				cb = f(&sb.d, 3) + " (" + pyPct(sb.spread) + ")"
				pb = f(fp(sb.d/float64(b)), 3)
			}
			cx, px, ex := "-", "-", "-"
			if b > 1 {
				sx, err := stats(rt, fmt.Sprintf("sweep_decode_x%d", b))
				if err != nil {
					return 0, err
				}
				if sx.status != "ok" {
					cx = sx.status
				} else {
					cx, px = f(&sx.d, 3)+" ("+pyPct(sx.spread)+")", f(fp(sx.d/float64(b)), 3)
					if sb.status == "ok" {
						ex = f(fp((sx.d-sb.d)/float64(b-1)), 3)
					}
				}
			}
			pl(w, fmt.Sprintf("| %d | %s | %s | %s | %s | %s | %s |", b, rt, cb, pb, cx, px, ex))
		}
	}
	return 0, nil
}
