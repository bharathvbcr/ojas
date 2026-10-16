package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Run from the repository root:
//
//	go test scripts/benchreport/main.go scripts/benchreport/main_test.go
//
// The want* goldens and fixtureRun at the end of this file were printed by
// the Python this tool replaced (bench/aggregate.py and bench/plotlib.py),
// run as the oracle on the same inputs before it was removed.

func TestPythonNumberText(t *testing.T) {
	// From Python 3.14: repr(v), format(v, "g"), format(v, ".3g").
	cases := []struct {
		v           float64
		repr, g, g3 string
	}{
		{0, "0.0", "0", "0"},
		{1, "1.0", "1", "1"},
		{100, "100.0", "100", "100"},
		{1000, "1000.0", "1000", "1e+03"},
		{1e6, "1000000.0", "1e+06", "1e+06"},
		{0.5, "0.5", "0.5", "0.5"},
		{0.0001, "0.0001", "0.0001", "0.0001"},
		{0.00001, "1e-05", "1e-05", "1e-05"},
		{123456, "123456.0", "123456", "1.23e+05"},
		{1234567, "1234567.0", "1.23457e+06", "1.23e+06"},
		{2.5e-05, "2.5e-05", "2.5e-05", "2.5e-05"},
		{12.3456, "12.3456", "12.3456", "12.3"},
		{99.95, "99.95", "99.95", "100"},
		{1e16, "1e+16", "1e+16", "1e+16"},
		{1.5e16, "1.5e+16", "1.5e+16", "1.5e+16"},
		{93.5, "93.5", "93.5", "93.5"},
		{190.39999999999998, "190.39999999999998", "190.4", "190"},
	}
	for _, c := range cases {
		if got := pyRepr(c.v); got != c.repr {
			t.Errorf("pyRepr(%v) = %q, want %q", c.v, got, c.repr)
		}
		if got := pyG(c.v, 6); got != c.g {
			t.Errorf("pyG(%v, 6) = %q, want %q", c.v, got, c.g)
		}
		if got := pyG(c.v, 3); got != c.g3 {
			t.Errorf("pyG(%v, 3) = %q, want %q", c.v, got, c.g3)
		}
	}
	if got := pyCapitalize("contended (other jobs running; the README says -2 supersedes it)"); got != "Contended (other jobs running; the readme says -2 supersedes it)" {
		t.Errorf("pyCapitalize = %q", got)
	}
	if got := prefix("abcé", 4); got != "abcé" {
		t.Errorf("prefix kept %q", got)
	}
	if got := prefix("abcéz", 4); got != "abcé" {
		t.Errorf("prefix cut %q", got)
	}
}

func writeRun(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestAggregateMatchesPythonOracle(t *testing.T) {
	dir := writeRun(t, fixtureRun)
	var out, errb bytes.Buffer
	code := runAggregate([]string{dir}, &out, &errb)
	got := strings.ReplaceAll(out.String(), dir, "RUN")
	if got != wantAggregate {
		t.Errorf("aggregate output differs from the Python oracle\n--- got\n%s\n--- want\n%s", got, wantAggregate)
	}
	// The fixture has a refused row, a crashed round and a torch failure, so
	// the run is incomplete and must say so through the exit status.
	if code != 1 || !strings.Contains(errb.String(), "incomplete") {
		t.Errorf("exit %d, stderr %q; want 1 and an incomplete-rows message", code, errb.String())
	}
}

func TestAggregateCompleteRunExitsZero(t *testing.T) {
	dir := writeRun(t, map[string]string{
		"round1/a.jsonl": `{"runtime":"ojas-metal","row":"add","status":"ok","min_ms":1,"median_ms":1}` + "\n" +
			`{"runtime":"torch-mps","row":"add","status":"ok","min_ms":2,"median_ms":2}` + "\n",
	})
	var out, errb bytes.Buffer
	if code := runAggregate([]string{dir}, &out, &errb); code != 0 {
		t.Fatalf("exit %d, stderr %q", code, errb.String())
	}
	if !strings.Contains(out.String(), "| add | 1.000 / 1.000 | 2.000 / 2.000 | 2.00x [2.00-2.00] |") {
		t.Errorf("unexpected summary:\n%s", out.String())
	}
}

func TestAggregateFailsLoudly(t *testing.T) {
	cases := map[string]map[string]string{
		"empty directory":       {"README": "no rounds here"},
		"malformed json":        {"round1/a.jsonl": "{not json}\n"},
		"record without row":    {"round1/a.jsonl": `{"runtime":"ojas-metal","status":"ok"}` + "\n"},
		"bad round directory":   {"roundX/a.jsonl": `{"runtime":"ojas-metal","row":"a","status":"ok","min_ms":1,"median_ms":1}` + "\n"},
		"load record lacks key": {"round1/a.jsonl": "", "load.jsonl": `{"round":1,"lane":"x"}` + "\n"},
		"ok row without median": {"round1/a.jsonl": `{"runtime":"ojas-metal","row":"a","status":"ok","min_ms":1}` + "\n" +
			`{"runtime":"torch-mps","row":"a","status":"ok","min_ms":1,"median_ms":1}` + "\n"},
	}
	for name, files := range cases {
		dir := writeRun(t, files)
		var out, errb bytes.Buffer
		if code := runAggregate([]string{dir}, &out, &errb); code == 0 {
			t.Errorf("%s: exit 0, stdout %q", name, out.String())
		}
	}
	var out, errb bytes.Buffer
	if code := runAggregate(nil, &out, &errb); code != 2 {
		t.Errorf("no arguments: exit %d, want 2", code)
	}
}

func TestChartsMatchPythonOracle(t *testing.T) {
	u := func(f float64) *float64 { return &f }
	rows := []ratioRow{
		{name: "fast row with a long name that runs past forty-six code points é", v: 0.42, lo: u(0.4), hi: u(0.5), mid: u(0.45)},
		{name: "slow & <noisy>", v: 3.3, lo: u(2.9), hi: u(4.1), mid: u(3.2), hollow: true},
		{name: "parity", v: 1.02},
		{name: "tiny", v: 0.9999, color: "#123456"},
	}
	if got, ok := ratioChart(rows, "Ratio & <title>", "sub", band); !ok || got != wantRatioSVG {
		t.Errorf("ratio chart differs from the Python oracle\n--- got\n%s", got)
	}
	groups := []group{
		{"first group", []*float64{u(1.5), nil, u(0), u(2500), u(0.003), u(7)}},
		{"second", []*float64{u(12), u(0.75), u(3), nil, u(99.95), u(123456)}},
	}
	ser := []series{{"alpha series", blue}, {"beta series name", orange}, {"gamma µs", green},
		{"delta delta delta delta", red}, {"epsilon", "#7b5ea7"}, {"zeta zeta zeta zeta zeta", "#8c564b"}}
	if got, ok := groupedChart(groups, ser, "Grouped", "values", "µs"); !ok || got != wantGroupedSVG {
		t.Errorf("grouped chart differs from the Python oracle\n--- got\n%s", got)
	}
	pts := func(xy ...float64) []pt {
		var out []pt
		for i := 0; i < len(xy); i += 2 {
			out = append(out, pt{u(xy[i]), u(xy[i+1])})
		}
		return out
	}
	lines := []lineSeries{{"min", blue, pts(1, 9.5, 2, 12.25, 5, 30, 3, 20)}, {"empty", red, pts(0, 1)},
		{"med", orange, pts(1, 11, 5, 38)}}
	if got, ok, err := lineChart(lines, "Lines", "sub", "threads", "µs", false, false); err != nil || !ok || got != wantLineSVG {
		t.Errorf("linear line chart differs from the Python oracle (err %v)\n--- got\n%s", err, got)
	}
	if got, ok, err := lineChart(lines, "Lines", "sub", "threads", "µs", true, true); err != nil || !ok || got != wantLogLineSVG {
		t.Errorf("log line chart differs from the Python oracle (err %v)\n--- got\n%s", err, got)
	}
	if _, ok := ratioChart(nil, "t", "s", band); ok {
		t.Error("an empty ratio chart was drawn")
	}
	if _, ok := groupedChart([]group{{"a", []*float64{nil, u(0)}}}, ser[:2], "t", "s", "ms"); ok {
		t.Error("a grouped chart with no values was drawn")
	}
	if _, _, err := lineChart([]lineSeries{{"x", blue, []pt{{nil, u(1)}}}}, "t", "s", "x", "y", true, true); err == nil {
		t.Error("a point with no x value was accepted")
	}
}

func TestParsePairedRecordsUnchartedRows(t *testing.T) {
	text := "3 rounds.\n\n### ojas-metal vs torch-mps\n\n| row | a | b | c |\n| :-- | --: | --: | :-- |\n" +
		"| add | 1.000 / 2.000 | 3.000 / 4.000 | 2.00x [1.50-2.50] | 1% / 2% | noisy - not quoted | - |\n" +
		"| mm | - | - | - | - | **r1 ojas refused: too big** | - |\n" +
		"| tiny | 1.000 / 400.0 | 1.000 / 2.000 | 0.01x [0.00-0.02] | 1% / 2% |  | - |\n"
	missing := map[string][]missRow{}
	ls, err := parsePaired(text, missing)
	if err != nil {
		t.Fatal(err)
	}
	rows := ls.rows["ojas-metal vs torch-mps"]
	if len(rows) != 2 || !rows[0].hollow || rows[0].lo == nil || *rows[0].lo != 1/2.5 {
		t.Fatalf("rows %+v", rows)
	}
	// Under 0.05x the printed ratio is too coarse: the medians' ratio is used, with no whisker.
	if rows[1].v != 200 || rows[1].lo != nil {
		t.Errorf("tiny row %+v", rows[1])
	}
	if m := missing["ojas-metal vs torch-mps"]; len(m) != 1 || m[0].row != "mm" || m[0].reason != "r1 ojas refused: too big**" { // Python's lstrip("* "): the closing ** stays

		t.Errorf("missing %+v", m)
	}
}

func summarize1(t *testing.T, cmd string, args ...string) (string, string, int) {
	t.Helper()
	var out, errb bytes.Buffer
	code := runSummary(summaries[cmd], args, &out, &errb)
	return out.String(), errb.String(), code
}

func TestSummariesMatchPythonOracle(t *testing.T) {
	dir := writeRun(t, summaryFixture)
	p := func(s string) string { return filepath.Join(dir, s) }
	cases := []struct {
		name, want string
		cmd        string
		args       []string
	}{
		{"bench-ops", wantBenchOps, "bench-ops", []string{"-parity", p("ops/parity.txt"), p("ops/r1.txt"), p("ops/r2.txt"), p("ops/t1.txt"), p("ops/t2.txt")}},
		{"cpu-ab", wantCPUAB, "cpu-ab", []string{p("hot")}},
		{"sweep-fit", wantSweepFit, "sweep-fit", []string{p("sweep")}},
		{"decode-ab", wantDecodeAB, "decode-ab", []string{p("dec"), "min_ms"}},
	}
	for _, c := range cases {
		got, stderr, code := summarize1(t, c.cmd, c.args...)
		got = strings.ReplaceAll(got, dir, "FX")
		if code != 0 || got != c.want {
			t.Errorf("%s: exit %d (%s); output differs from the Python oracle\n--- got\n%s\n--- want\n%s", c.name, code, stderr, got, c.want)
		}
	}
}

func TestPySumIsCompensated(t *testing.T) {
	tenths := make([]pyNumber, 10)
	for i := range tenths {
		tenths[i] = pyNumber{f: 0.1}
	}
	if got := pySum(tenths); got != wantSumTenths {
		t.Errorf("sum of ten 0.1 = %v, Python says %v", got, wantSumTenths)
	}
	if got := pySum([]pyNumber{{f: 1e16}, {f: 1}, {f: -1e16}}); got != wantSumCancel {
		t.Errorf("sum(1e16, 1, -1e16) = %v, Python says %v", got, wantSumCancel)
	}
	if got := pySum([]pyNumber{{isInt: true}, {isInt: true, i: 3}}); got != 3 {
		t.Errorf("int sum = %v", got)
	}
}

func TestSummariesFailLoudly(t *testing.T) {
	cases := []struct {
		name  string
		files map[string]string
		cmd   string
		args  func(dir string) []string
	}{
		{"cpu-ab-linux cell without its after lane", map[string]string{"ops-1-base.txt": "OJAS_OP op=a dir=fwd threads=1 min_ms=1\n"},
			"cpu-ab-linux", func(d string) []string { return []string{d} }},
		{"bpe without the base lane", map[string]string{"r1_a.txt": "digest=ab tiktoken_parity=ok documents_matched=1/1\n| encode_docs | 1 | x | 2 | 3 |\n"},
			"bpe", func(d string) []string { return []string{d, "base"} }},
		{"bpe file without a digest", map[string]string{"r1_base.txt": "| encode_docs | 1 | x | 2 | 3 |\n"},
			"bpe", func(d string) []string { return []string{d, "base"} }},
		{"attn-ab round without its control", map[string]string{"ab.txt": "AB round=1 tree=old backend=metal shape=s window=full bwd_min=1 fwd_min=1\n" +
			"AB round=1 tree=new backend=metal shape=s window=full bwd_min=1 fwd_min=1\n"},
			"attn-ab", func(d string) []string { return []string{filepath.Join(d, "ab.txt")} }},
		{"decode-ab blank line", map[string]string{"round1/metal-before.jsonl": "\n"},
			"decode-ab", func(d string) []string { return []string{d} }},
		{"sweep-fit bad record", map[string]string{"round1/a.jsonl": `{"runtime":"x"}` + "\n"},
			"sweep-fit", func(d string) []string { return []string{d} }},
		{"bench-ops op line without threads", map[string]string{"r.txt": "OJAS_OP op=a dir=fwd shape=[1] min_ms=1\n"},
			"bench-ops", func(d string) []string { return []string{filepath.Join(d, "r.txt")} }},
	}
	for _, c := range cases {
		dir := writeRun(t, c.files)
		out, _, code := summarize1(t, c.cmd, c.args(dir)...)
		if code != 1 || out != "" {
			t.Errorf("%s: exit %d, stdout %q; want 1 and nothing on stdout", c.name, code, out)
		}
	}
	if _, _, code := summarize1(t, "bpe", "only-one-arg"); code != 2 {
		t.Errorf("bpe with one argument: exit %d, want 2", code)
	}
	// A record that is neither ok nor info is printed and fails the run.
	dir := writeRun(t, map[string]string{
		"round1/metal-before.jsonl": `{"row":"gen_prefill_p32","status":"ok","min_ms":1,"median_ms":1}` + "\n" + `{"row":"gen_x","status":"crash"}` + "\n",
		"round1/metal-after.jsonl":  `{"row":"gen_prefill_p32","status":"ok","min_ms":1,"median_ms":1}` + "\n",
	})
	out, stderr, code := summarize1(t, "decode-ab", dir)
	if code != 1 || !strings.Contains(out, `PROBLEM metal before 1 {"row":"gen_x","status":"crash"}`) || !strings.Contains(stderr, "neither ok nor info") {
		t.Errorf("decode-ab problem record: exit %d\n%s\n%s", code, out, stderr)
	}
}

// The committed charts and page are what the committed results produce.
func TestCommittedPlotsAreCurrent(t *testing.T) {
	root := filepath.Join("..", "..")
	if _, err := os.Stat(filepath.Join(root, "bench", "results")); err != nil {
		t.Skip("not run from scripts/benchreport inside the repository")
	}
	var out, errb bytes.Buffer
	if code := runPlots(root, true, &out, &errb); code != 0 {
		t.Fatalf("plots -check exit %d:\n%s", code, errb.String())
	}
}

// Code below was printed by a throwaway Python script that ran bench/aggregate.py
// and bench/plotlib.py (the oracle) before they were removed.

var fixtureRun = map[string]string{
	"round1/metal.jsonl": "{\"runtime\":\"ojas-metal\",\"row\":\"_device\",\"status\":\"info\"}\n{\"runtime\":\"ojas-metal\",\"row\":\"add\",\"status\":\"ok\",\"min_ms\":1.5,\"median_ms\":2,\"parity_max_abs\":1e-6,\"parity_rel\":2.5e-7}\n{\"runtime\":\"ojas-metal\",\"row\":\"_load\",\"for\":\"add\",\"when\":\"before\",\"load\":\"3.10, 4.00 5.00\",\"gpu_util_pct\":45}\n{\"runtime\":\"ojas-metal\",\"row\":\"mm\",\"status\":\"refused\",\"detail\":\"too big\"}\n{\"runtime\":\"ojas-metal\",\"row\":\"gen_prefill_p8\",\"status\":\"ok\",\"min_ms\":3,\"median_ms\":4.0}\n{\"runtime\":\"ojas-metal\",\"row\":\"gen_greedy_p8_n5\",\"status\":\"ok\",\"min_ms\":10,\"median_ms\":12.5}\n{\"runtime\":\"ojas-metal\",\"row\":\"_gen_traffic\",\"for\":\"gen_greedy_p8_n5\",\"status\":\"info\",\"step_uploads\":2,\"step_upload_bytes\":64,\"step_readback_bytes\":4.0}\n{\"runtime\":\"ojas-metal\",\"row\":\"_gen_ids\",\"for\":\"gen_greedy_p8_n5\",\"status\":\"info\",\"ids_equal\":4,\"of\":4,\"common_prefix\":4}\n\n",
	"round1/torch.jsonl": "{\"runtime\":\"torch-mps\",\"row\":\"add\",\"status\":\"ok\",\"min_ms\":3.0,\"median_ms\":3.5}\n{\"runtime\":\"torch-mps\",\"row\":\"add_bf16\",\"status\":\"ok\",\"min_ms\":2.0,\"median_ms\":2.5}\n{\"runtime\":\"torch-mps\",\"row\":\"mm\",\"status\":\"ok\",\"min_ms\":20,\"median_ms\":22}\n{\"runtime\":\"torch-mps\",\"row\":\"only_torch\",\"status\":\"ok\",\"min_ms\":1,\"median_ms\":1}\n{\"runtime\":\"torch-mps\",\"row\":\"gen_prefill_p8\",\"status\":\"ok\",\"min_ms\":2,\"median_ms\":2.5}\n{\"runtime\":\"torch-mps\",\"row\":\"gen_greedy_p8_n5\",\"status\":\"ok\",\"min_ms\":8,\"median_ms\":9.0}\n",
	"round2/metal.jsonl": "{\"runtime\":\"ojas-metal\",\"row\":\"add\",\"status\":\"ok\",\"min_ms\":1.4,\"median_ms\":2.6}\r\n{\"runtime\":\"ojas-metal\",\"row\":\"_process\",\"status\":\"crash\",\"exit\":139}\r\n{\"runtime\":\"ojas-metal\",\"row\":\"_load\",\"for\":\"add\",\"when\":\"after\",\"load\":\"bad\",\"gpu_util_pct\":50.5}\r\n{\"runtime\":\"ojas-metal\",\"row\":\"gen_prefill_p8\",\"status\":\"ok\",\"min_ms\":3.2,\"median_ms\":4.4}\r\n{\"runtime\":\"ojas-metal\",\"row\":\"gen_greedy_p8_n5\",\"status\":\"ok\",\"min_ms\":11,\"median_ms\":13}\r\n{\"runtime\":\"ojas-metal\",\"row\":\"_gen_readback\",\"for\":\"vocab\",\"status\":\"info\",\"bytes\":1024,\"to_host_us\":3.5,\"read_bytes_us\":1.25,\"decode_pass_us\":2.25,\"iters\":50}\r\n",
	"round2/torch.jsonl": "{\"runtime\":\"torch-mps\",\"row\":\"add\",\"status\":\"ok\",\"min_ms\":3.1,\"median_ms\":3.4}\n{\"runtime\":\"torch-mps\",\"row\":\"add_bf16\",\"status\":\"failed\",\"detail\":\"bf16 unsupported\"}\n{\"runtime\":\"torch-mps\",\"row\":\"mm\",\"status\":\"ok\",\"min_ms\":21,\"median_ms\":23}\n{\"runtime\":\"torch-mps\",\"row\":\"gen_prefill_p8\",\"status\":\"ok\",\"min_ms\":2.1,\"median_ms\":2.6}\n{\"runtime\":\"torch-mps\",\"row\":\"gen_greedy_p8_n5\",\"status\":\"ok\",\"min_ms\":8.5,\"median_ms\":9.5}\n",
	"env.txt":            "macOS 27\r\nrustc 1.97\n\n",
	"load.jsonl":         "{\"round\":1,\"lane\":\"metal\",\"when\":\"start\",\"time\":\"08:00\",\"load\":\"1.0 2.0 3.0\",\"gpu_util_pct\":12}\n{\"round\":2,\"lane\":\"torch\",\"when\":\"end\",\"time\":\"08:05\",\"load\":\"1.5 2.0 3.0\",\"gpu_util_pct\":12.0}\n",
}

const wantAggregate = "# Paired GPU-vs-torch summary: RUN\n\n2 rounds. ratio = torch median / ojas median per round (> 1: ojas faster). Spread = max/min of that side's per-round medians.\n\n```\nmacOS 27\nrustc 1.97\n```\n\n### ojas-metal vs torch-mps\n\n| row | ojas min / median ms | torch min / median ms | median ratio [min-max] | spread ojas / torch | flag | parity max abs (rel) |\n| :-- | --: | --: | :-- | :-- | :-- | :-- |\n| add | 1.400 / 2.300 | 3.000 / 3.450 | 1.53x [1.31-1.75] | 30% / 3% | noisy - not quoted | 1.00e-06 (2.5e-07) |\n| add vs add_bf16 | 1.500 / 2.000 | 2.000 / 2.500 | 1.25x [1.25-1.25] | 0% / 0% | partial (1 rounds): r2 torch: failed bf16 unsupported | 1.00e-06 (2.5e-07) |\n| mm | - | - | - | - | **r1 ojas refused: too big; r2 ojas: crash exit 139** | - |\n| gen_prefill_p8 | 3.000 / 4.200 | 2.000 / 2.550 | 0.61x [0.59-0.62] | 10% / 4% | noisy - not quoted | - |\n| gen_greedy_p8_n5 | 10.00 / 12.75 | 8.000 / 9.250 | 0.73x [0.72-0.73] | 4% / 6% |  | - |\n| only_torch | - | - | - | - | **NO DATA: row absent from ojas-metal** | - |\n\n### Ranked: where ojas is slowest relative to torch (lowest ratio first)\n\n| rank | backend | row | median ratio [min-max] | ojas median ms | torch median ms | flag |\n| --: | :-- | :-- | :-- | --: | --: | :-- |\n| 1 | ojas-metal | gen_prefill_p8 | 0.61x [0.59-0.62] | 4.200 | 2.550 | noisy |\n| 2 | ojas-metal | gen_greedy_p8_n5 | 0.73x [0.72-0.73] | 12.75 | 9.250 |  |\n| 3 | ojas-metal | add vs add_bf16 | 1.25x [1.25-1.25] | 2.000 | 2.500 |  |\n| 4 | ojas-metal | add | 1.53x [1.31-1.75] | 2.300 | 3.450 | noisy |\n\n### Machine load around each runtime block\n\n| round | lane | when | time | load avg (1, 5, 15 min) | GPU Device Utilization % |\n| --: | :-- | :-- | :-- | :-- | --: |\n| 1 | metal | start | 08:00 | 1.0 2.0 3.0 | 12 |\n| 2 | torch | end | 08:05 | 1.5 2.0 3.0 | 12.0 |\n\n### Decode (gen_ rows)\n\nPer-token ms = (decode median - prefill median) / (N - 1), median over rounds, each decode row against its model's prefill row. Traffic is per decode step, from DeviceDecoder's counters (uploads / bytes up / bytes read back).\n\n| runtime | row | prefill median ms | decode ms / token | tokens / s | rounds | step traffic | greedy ids equal to torch |\n| :-- | :-- | --: | --: | --: | --: | :-- | :-- |\n| torch-mps | gen_greedy_p8_n5 | 2.550 | 1.675 | 597.0 | 2 | - | - |\n| ojas-metal | gen_greedy_p8_n5 | 4.200 | 2.138 | 467.8 | 2 | 2 / 64 B / 4 B | 4/4 (prefix 4) |\n\nReading one `[1, vocab]` logit row back (last round, min of `iters`): `Tensor::to_host` against `DeviceBuffer::read_bytes` alone; the difference is the transient decode pass a read into a caller's buffer would save.\n\n| runtime | bytes | to_host us | read_bytes us | decode pass us | iters |\n| :-- | --: | --: | --: | --: | --: |\n| ojas-metal | 1024 | 3.50 | 1.25 | 2.25 | 50 |\n\n### Load recorded before and after every row\n\n| round | lane | samples | 1-min load min-max | GPU util % min / median / max |\n| --: | :-- | --: | :-- | :-- |\n| 1 | metal | 1 | 3.1-3.1 | 45 / 45 / 45 |\n| 2 | metal | 0 | - | 50.5 / 50 / 50.5 |\n"

const wantRatioSVG = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"920\" height=\"208\" viewBox=\"0 0 920 208\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"12\">\n<rect width=\"100%\" height=\"100%\" fill=\"#fff\"/>\n<text x=\"16\" y=\"24\" font-size=\"15\" font-weight=\"bold\" fill=\"#111\">Ratio &amp; &lt;title&gt;</text>\n<text x=\"16\" y=\"44\" fill=\"#555\">sub</text>\n<text x=\"16\" y=\"62\" fill=\"#555\">Lower is faster. Grey band = within 10%. Hollow bar = run marked noisy. Whisker = per-round min..max.</text>\n<rect x=\"552.4\" y=\"72\" width=\"33.9\" height=\"86\" fill=\"#eee\"/>\n<line x1=\"446.0\" x2=\"446.0\" y1=\"72\" y2=\"158\" stroke=\"#ddd\" stroke-width=\"1\"/>\n<text x=\"446.0\" y=\"174\" text-anchor=\"middle\" fill=\"#444\">0.5x</text>\n<line x1=\"569.3\" x2=\"569.3\" y1=\"72\" y2=\"158\" stroke=\"#222\" stroke-width=\"1.5\"/>\n<text x=\"569.3\" y=\"174\" text-anchor=\"middle\" fill=\"#444\">1x</text>\n<line x1=\"692.6\" x2=\"692.6\" y1=\"72\" y2=\"158\" stroke=\"#ddd\" stroke-width=\"1\"/>\n<text x=\"692.6\" y=\"174\" text-anchor=\"middle\" fill=\"#444\">2x</text>\n<line x1=\"855.6\" x2=\"855.6\" y1=\"72\" y2=\"158\" stroke=\"#ddd\" stroke-width=\"1\"/>\n<text x=\"855.6\" y=\"174\" text-anchor=\"middle\" fill=\"#444\">5x</text>\n<text x=\"292\" y=\"92\" text-anchor=\"end\" fill=\"#222\">fast row with a long name that runs past forty</text>\n<rect x=\"415.0\" y=\"81\" width=\"154.3\" height=\"13\" fill=\"#2a9d8f\"/>\n<line x1=\"406.3\" x2=\"446.0\" y1=\"87.5\" y2=\"87.5\" stroke=\"#222\" stroke-width=\"1\"/>\n<line x1=\"406.3\" x2=\"406.3\" y1=\"84.5\" y2=\"90.5\" stroke=\"#222\" stroke-width=\"1\"/>\n<line x1=\"446.0\" x2=\"446.0\" y1=\"84.5\" y2=\"90.5\" stroke=\"#222\" stroke-width=\"1\"/>\n<circle cx=\"427.3\" cy=\"87.5\" r=\"3.2\" fill=\"#fff\" stroke=\"#222\" stroke-width=\"1.2\"/>\n<text x=\"401.3\" y=\"92\" text-anchor=\"end\" fill=\"#333\">0.42</text>\n<text x=\"292\" y=\"112\" text-anchor=\"end\" fill=\"#222\">slow &amp; &lt;noisy&gt;</text>\n<rect x=\"569.3\" y=\"101\" width=\"212.4\" height=\"13\" fill=\"#fff\" stroke=\"#d1495b\" stroke-width=\"1.4\"/>\n<line x1=\"758.7\" x2=\"820.3\" y1=\"107.5\" y2=\"107.5\" stroke=\"#222\" stroke-width=\"1\"/>\n<line x1=\"758.7\" x2=\"758.7\" y1=\"104.5\" y2=\"110.5\" stroke=\"#222\" stroke-width=\"1\"/>\n<line x1=\"820.3\" x2=\"820.3\" y1=\"104.5\" y2=\"110.5\" stroke=\"#222\" stroke-width=\"1\"/>\n<circle cx=\"776.2\" cy=\"107.5\" r=\"3.2\" fill=\"#fff\" stroke=\"#222\" stroke-width=\"1.2\"/>\n<text x=\"825.3\" y=\"112\" text-anchor=\"start\" fill=\"#333\">3.30</text>\n<text x=\"292\" y=\"132\" text-anchor=\"end\" fill=\"#222\">parity</text>\n<rect x=\"569.3\" y=\"121\" width=\"3.5\" height=\"13\" fill=\"#9aa0a6\"/>\n<text x=\"577.8\" y=\"132\" text-anchor=\"start\" fill=\"#333\">1.02</text>\n<text x=\"292\" y=\"152\" text-anchor=\"end\" fill=\"#222\">tiny</text>\n<rect x=\"569.3\" y=\"141\" width=\"1.0\" height=\"13\" fill=\"#123456\"/>\n<text x=\"564.3\" y=\"152\" text-anchor=\"end\" fill=\"#333\">1.00</text>\n</svg>"

const wantGroupedSVG = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"890\" height=\"284\" viewBox=\"0 0 890 284\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"12\">\n<rect width=\"100%\" height=\"100%\" fill=\"#fff\"/>\n<text x=\"16\" y=\"24\" font-size=\"15\" font-weight=\"bold\" fill=\"#111\">Grouped</text>\n<text x=\"16\" y=\"44\" fill=\"#555\">values</text>\n<circle cx=\"22\" cy=\"60\" r=\"5\" fill=\"#3d6fb6\"/><text x=\"33\" y=\"64\" fill=\"#333\">alpha series</text>\n<circle cx=\"140.8\" cy=\"60\" r=\"5\" fill=\"#e08e2b\"/><text x=\"151.8\" y=\"64\" fill=\"#333\">beta series name</text>\n<circle cx=\"285.20000000000005\" cy=\"60\" r=\"5\" fill=\"#2a9d8f\"/><text x=\"296.20000000000005\" y=\"64\" fill=\"#333\">gamma µs</text>\n<circle cx=\"378.40000000000003\" cy=\"60\" r=\"5\" fill=\"#d1495b\"/><text x=\"389.40000000000003\" y=\"64\" fill=\"#333\">delta delta delta delta</text>\n<circle cx=\"567.6\" cy=\"60\" r=\"5\" fill=\"#7b5ea7\"/><text x=\"578.6\" y=\"64\" fill=\"#333\">epsilon</text>\n<circle cx=\"654.4000000000001\" cy=\"60\" r=\"5\" fill=\"#8c564b\"/><text x=\"665.4000000000001\" y=\"64\" fill=\"#333\">zeta zeta zeta zeta zeta</text>\n<line x1=\"344.2\" x2=\"344.2\" y1=\"78\" y2=\"234\" stroke=\"#ddd\"/>\n<text x=\"344.2\" y=\"250\" text-anchor=\"middle\" fill=\"#444\">0.01</text>\n<line x1=\"409.4\" x2=\"409.4\" y1=\"78\" y2=\"234\" stroke=\"#ddd\"/>\n<text x=\"409.4\" y=\"250\" text-anchor=\"middle\" fill=\"#444\">0.1</text>\n<line x1=\"474.6\" x2=\"474.6\" y1=\"78\" y2=\"234\" stroke=\"#ddd\"/>\n<text x=\"474.6\" y=\"250\" text-anchor=\"middle\" fill=\"#444\">1</text>\n<line x1=\"539.9\" x2=\"539.9\" y1=\"78\" y2=\"234\" stroke=\"#ddd\"/>\n<text x=\"539.9\" y=\"250\" text-anchor=\"middle\" fill=\"#444\">10</text>\n<line x1=\"605.1\" x2=\"605.1\" y1=\"78\" y2=\"234\" stroke=\"#ddd\"/>\n<text x=\"605.1\" y=\"250\" text-anchor=\"middle\" fill=\"#444\">100</text>\n<line x1=\"670.3\" x2=\"670.3\" y1=\"78\" y2=\"234\" stroke=\"#ddd\"/>\n<text x=\"670.3\" y=\"250\" text-anchor=\"middle\" fill=\"#444\">1000</text>\n<line x1=\"735.5\" x2=\"735.5\" y1=\"78\" y2=\"234\" stroke=\"#ddd\"/>\n<text x=\"735.5\" y=\"250\" text-anchor=\"middle\" fill=\"#444\">10000</text>\n<line x1=\"800.7\" x2=\"800.7\" y1=\"78\" y2=\"234\" stroke=\"#ddd\"/>\n<text x=\"800.7\" y=\"250\" text-anchor=\"middle\" fill=\"#444\">100000</text>\n<text x=\"560.0\" y=\"268\" text-anchor=\"middle\" fill=\"#444\">µs (log axis)</text>\n<text x=\"292\" y=\"121.0\" text-anchor=\"end\" fill=\"#222\">first group</text>\n<line x1=\"300\" x2=\"486.1\" y1=\"89.0\" y2=\"89.0\" stroke=\"#3d6fb6\" stroke-opacity=\"0.35\"/>\n<circle cx=\"486.1\" cy=\"89.0\" r=\"4.2\" fill=\"#3d6fb6\"/>\n<text x=\"494.1\" y=\"93\" fill=\"#333\" font-size=\"10\">1.5</text>\n<line x1=\"300\" x2=\"696.2\" y1=\"122.0\" y2=\"122.0\" stroke=\"#d1495b\" stroke-opacity=\"0.35\"/>\n<circle cx=\"696.2\" cy=\"122.0\" r=\"4.2\" fill=\"#d1495b\"/>\n<text x=\"704.2\" y=\"126\" fill=\"#333\" font-size=\"10\">2.5e+03</text>\n<line x1=\"300\" x2=\"310.1\" y1=\"133.0\" y2=\"133.0\" stroke=\"#7b5ea7\" stroke-opacity=\"0.35\"/>\n<circle cx=\"310.1\" cy=\"133.0\" r=\"4.2\" fill=\"#7b5ea7\"/>\n<text x=\"318.1\" y=\"137\" fill=\"#333\" font-size=\"10\">0.003</text>\n<line x1=\"300\" x2=\"529.8\" y1=\"144.0\" y2=\"144.0\" stroke=\"#8c564b\" stroke-opacity=\"0.35\"/>\n<circle cx=\"529.8\" cy=\"144.0\" r=\"4.2\" fill=\"#8c564b\"/>\n<text x=\"537.8\" y=\"148\" fill=\"#333\" font-size=\"10\">7</text>\n<text x=\"292\" y=\"196.0\" text-anchor=\"end\" fill=\"#222\">second</text>\n<line x1=\"300\" x2=\"545.0\" y1=\"164.0\" y2=\"164.0\" stroke=\"#3d6fb6\" stroke-opacity=\"0.35\"/>\n<circle cx=\"545.0\" cy=\"164.0\" r=\"4.2\" fill=\"#3d6fb6\"/>\n<text x=\"553.0\" y=\"168\" fill=\"#333\" font-size=\"10\">12</text>\n<line x1=\"300\" x2=\"466.5\" y1=\"175.0\" y2=\"175.0\" stroke=\"#e08e2b\" stroke-opacity=\"0.35\"/>\n<circle cx=\"466.5\" cy=\"175.0\" r=\"4.2\" fill=\"#e08e2b\"/>\n<text x=\"474.5\" y=\"179\" fill=\"#333\" font-size=\"10\">0.75</text>\n<line x1=\"300\" x2=\"505.8\" y1=\"186.0\" y2=\"186.0\" stroke=\"#2a9d8f\" stroke-opacity=\"0.35\"/>\n<circle cx=\"505.8\" cy=\"186.0\" r=\"4.2\" fill=\"#2a9d8f\"/>\n<text x=\"513.8\" y=\"190\" fill=\"#333\" font-size=\"10\">3</text>\n<line x1=\"300\" x2=\"605.1\" y1=\"208.0\" y2=\"208.0\" stroke=\"#7b5ea7\" stroke-opacity=\"0.35\"/>\n<circle cx=\"605.1\" cy=\"208.0\" r=\"4.2\" fill=\"#7b5ea7\"/>\n<text x=\"613.1\" y=\"212\" fill=\"#333\" font-size=\"10\">100</text>\n<line x1=\"300\" x2=\"806.7\" y1=\"219.0\" y2=\"219.0\" stroke=\"#8c564b\" stroke-opacity=\"0.35\"/>\n<circle cx=\"806.7\" cy=\"219.0\" r=\"4.2\" fill=\"#8c564b\"/>\n<text x=\"814.7\" y=\"223\" fill=\"#333\" font-size=\"10\">1.23e+05</text>\n</svg>"

const wantLineSVG = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"720\" height=\"380\" viewBox=\"0 0 720 380\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"12\">\n<rect width=\"100%\" height=\"100%\" fill=\"#fff\"/>\n<text x=\"16\" y=\"24\" font-size=\"15\" font-weight=\"bold\" fill=\"#111\">Lines</text>\n<text x=\"16\" y=\"44\" fill=\"#555\">sub</text>\n<line x1=\"70\" x2=\"570\" y1=\"305.6\" y2=\"305.6\" stroke=\"#e5e5e5\"/><text x=\"64\" y=\"309.6\" text-anchor=\"end\" fill=\"#444\">10</text>\n<line x1=\"70\" x2=\"570\" y1=\"245.4\" y2=\"245.4\" stroke=\"#e5e5e5\"/><text x=\"64\" y=\"249.4\" text-anchor=\"end\" fill=\"#444\">20</text>\n<line x1=\"70.0\" x2=\"70.0\" y1=\"74\" y2=\"320\" stroke=\"#eee\"/><text x=\"70.0\" y=\"336\" text-anchor=\"middle\" fill=\"#444\">1</text>\n<line x1=\"195.0\" x2=\"195.0\" y1=\"74\" y2=\"320\" stroke=\"#eee\"/><text x=\"195.0\" y=\"336\" text-anchor=\"middle\" fill=\"#444\">2</text>\n<line x1=\"570.0\" x2=\"570.0\" y1=\"74\" y2=\"320\" stroke=\"#eee\"/><text x=\"570.0\" y=\"336\" text-anchor=\"middle\" fill=\"#444\">5</text>\n<text x=\"320.0\" y=\"368\" text-anchor=\"middle\" fill=\"#333\">threads</text>\n<text x=\"16\" y=\"66\" fill=\"#333\">µs</text>\n<path d=\"M70.0,308.6 L195.0,292.0 L320.0,245.4 L570.0,185.3\" fill=\"none\" stroke=\"#3d6fb6\" stroke-width=\"2\"/>\n<circle cx=\"70.0\" cy=\"308.6\" r=\"2.6\" fill=\"#3d6fb6\"/>\n<circle cx=\"195.0\" cy=\"292.0\" r=\"2.6\" fill=\"#3d6fb6\"/>\n<circle cx=\"320.0\" cy=\"245.4\" r=\"2.6\" fill=\"#3d6fb6\"/>\n<circle cx=\"570.0\" cy=\"185.3\" r=\"2.6\" fill=\"#3d6fb6\"/>\n<rect x=\"582\" y=\"80\" width=\"10\" height=\"10\" fill=\"#3d6fb6\"/><text x=\"597\" y=\"89\" fill=\"#333\">min</text>\n<path d=\"M70.0,299.5 L570.0,137.1\" fill=\"none\" stroke=\"#e08e2b\" stroke-width=\"2\"/>\n<circle cx=\"70.0\" cy=\"299.5\" r=\"2.6\" fill=\"#e08e2b\"/>\n<circle cx=\"570.0\" cy=\"137.1\" r=\"2.6\" fill=\"#e08e2b\"/>\n<rect x=\"582\" y=\"116\" width=\"10\" height=\"10\" fill=\"#e08e2b\"/><text x=\"597\" y=\"125\" fill=\"#333\">med</text>\n</svg>"

const wantLogLineSVG = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"720\" height=\"380\" viewBox=\"0 0 720 380\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"12\">\n<rect width=\"100%\" height=\"100%\" fill=\"#fff\"/>\n<text x=\"16\" y=\"24\" font-size=\"15\" font-weight=\"bold\" fill=\"#111\">Lines</text>\n<text x=\"16\" y=\"44\" fill=\"#555\">sub</text>\n<line x1=\"70\" x2=\"570\" y1=\"284.1\" y2=\"284.1\" stroke=\"#e5e5e5\"/><text x=\"64\" y=\"288.1\" text-anchor=\"end\" fill=\"#444\">10</text>\n<line x1=\"70\" x2=\"570\" y1=\"193.3\" y2=\"193.3\" stroke=\"#e5e5e5\"/><text x=\"64\" y=\"197.3\" text-anchor=\"end\" fill=\"#444\">20</text>\n<line x1=\"70.0\" x2=\"70.0\" y1=\"74\" y2=\"320\" stroke=\"#eee\"/><text x=\"70.0\" y=\"336\" text-anchor=\"middle\" fill=\"#444\">1</text>\n<line x1=\"285.3\" x2=\"285.3\" y1=\"74\" y2=\"320\" stroke=\"#eee\"/><text x=\"285.3\" y=\"336\" text-anchor=\"middle\" fill=\"#444\">2</text>\n<line x1=\"570.0\" x2=\"570.0\" y1=\"74\" y2=\"320\" stroke=\"#eee\"/><text x=\"570.0\" y=\"336\" text-anchor=\"middle\" fill=\"#444\">5</text>\n<text x=\"320.0\" y=\"368\" text-anchor=\"middle\" fill=\"#333\">threads</text>\n<text x=\"16\" y=\"66\" fill=\"#333\">µs</text>\n<path d=\"M70.0,290.8 L285.3,257.5 L411.3,193.3 L570.0,140.2\" fill=\"none\" stroke=\"#3d6fb6\" stroke-width=\"2\"/>\n<circle cx=\"70.0\" cy=\"290.8\" r=\"2.6\" fill=\"#3d6fb6\"/>\n<circle cx=\"285.3\" cy=\"257.5\" r=\"2.6\" fill=\"#3d6fb6\"/>\n<circle cx=\"411.3\" cy=\"193.3\" r=\"2.6\" fill=\"#3d6fb6\"/>\n<circle cx=\"570.0\" cy=\"140.2\" r=\"2.6\" fill=\"#3d6fb6\"/>\n<rect x=\"582\" y=\"80\" width=\"10\" height=\"10\" fill=\"#3d6fb6\"/><text x=\"597\" y=\"89\" fill=\"#333\">min</text>\n<path d=\"M70.0,271.6 L570.0,109.2\" fill=\"none\" stroke=\"#e08e2b\" stroke-width=\"2\"/>\n<circle cx=\"70.0\" cy=\"271.6\" r=\"2.6\" fill=\"#e08e2b\"/>\n<circle cx=\"570.0\" cy=\"109.2\" r=\"2.6\" fill=\"#e08e2b\"/>\n<rect x=\"582\" y=\"116\" width=\"10\" height=\"10\" fill=\"#e08e2b\"/><text x=\"597\" y=\"125\" fill=\"#333\">med</text>\n</svg>"

var summaryFixture = map[string]string{
	"ops/r1.txt":                    "OJAS_OP op=add dir=fwd shape=[4,8] threads=1 min_ms=1.0\nOJAS_OP op=add dir=fwd shape=[4,8] threads=6 min_ms=0.5\nOJAS_OP op=sdpa dir=fwd shape=[1,2,3] threads=1 min_ms=12.5\nOJAS_OP op=clip dir=step shape=[9] threads=1 min_ms=0.25\nOJAS_OP op=block dir=fwd shape=[b] threads=1 min_ms=20.0\nOJAS_OP op=block dir=step shape=[b] threads=1 min_ms=45.0\nOJAS_OP op=rms_norm dir=fwd shape=[r] threads=1 min_ms=0.1\nOJAS_OP op=rms_norm dir=bwd shape=[r] threads=1 min_ms=0.2\nOJAS_OP op=add dir=bwd shape=[4,8] threads=1 min_ms=0.3\nnoise line\n",
	"ops/r2.txt":                    "OJAS_OP op=add dir=fwd shape=[4,8] threads=1 min_ms=1.3\nOJAS_OP op=add dir=fwd shape=[4,8] threads=6 min_ms=0.55\nOJAS_OP op=sdpa dir=fwd shape=[1,2,3] threads=1 min_ms=11.0\nOJAS_OP op=clip dir=step shape=[9] threads=1 min_ms=0.2\n",
	"ops/t1.txt":                    "TORCH_OP op=add dir=fwd variant=default min_ms=2.0\nTORCH_OP op=sdpa dir=fwd variant=math min_ms=10.0\nTORCH_OP op=sdpa dir=fwd variant=default min_ms=9.0\nTORCH_OP op=clip dir=step variant=default min_ms=0.0\nTORCH_OP op=block dir=fwd variant=math min_ms=25.0\nTORCH_OP op=block dir=step variant=math min_ms=50.0\n",
	"ops/t2.txt":                    "TORCH_OP op=add dir=fwd variant=default min_ms=3.0\nTORCH_OP op=sdpa dir=fwd variant=math min_ms=13.0\n",
	"ops/parity.txt":                "PARITY op=add dir=fwd variant=default out=y ok=True norm_err=1.5e-07 tol=1e-5\nPARITY op=add dir=fwd variant=default out=z ok=True norm_err=3e-08 tol=1e-5\nPARITY op=sdpa dir=fwd variant=math out=o ok=False norm_err=2e-3 tol=1e-4\nPARITY op=clip dir=step variant=default out=norm ok=False norm_err=0.004 tol=1e-4\nPARITY op=clip dir=step variant=default out=g ok=False norm_err=0.00401 tol=1e-4\nPARITY op=clip dir=step variant=default out=h ok=False norm_err=0.5 tol=1e-4\nPARITY op=block dir=fwd variant=math out=y ok=False tol=1e-3\nPARITY_F64 op=sdpa dir=fwd out=o ojas_ok=False ojas_err=1e-2 torch_err=1e-6\nPARITY_F64 op=clip dir=step out=norm ojas_ok=True ojas_err=1e-8 torch_err=0.004\n",
	"hot/ops-1-base.txt":            "OJAS_OP op=mul dir=fwd min_ms=2.0\nOJAS_OP op=adamw_50304x768 dir=step min_ms=8.0\n",
	"hot/ops-3-base.txt":            "OJAS_OP op=mul dir=fwd min_ms=2.5\n",
	"hot/ops-1-after.txt":           "OJAS_OP op=mul dir=fwd min_ms=1.0\nOJAS_OP op=adamw_50304x768 dir=step min_ms=4.0\nOJAS_OP op=conv dir=bwd variant=im2col min_ms=3.0\n",
	"hot/ops-2-after.txt":           "OJAS_OP op=mul dir=fwd min_ms=1.2\n",
	"hot/ops-3-after.txt":           "OJAS_OP op=mul dir=fwd min_ms=0.9\n",
	"hot/ops-1-torch.txt":           "TORCH_OP op=mul dir=fwd variant=default min_ms=1.5\nTORCH_OP op=adamw_50304x768 dir=step variant=fused min_ms=5.0\n",
	"hot/tape-1-after.txt":          "junk OJAS_TAPE\nOJAS_TAPE row=fanin min_ms=4.0\n",
	"hot/tape-1-base.txt":           "OJAS_TAPE row=fanin min_ms=5.0\n",
	"hot/notes.txt":                 "ignored\n",
	"sweep/round1/m.jsonl":          "{\"runtime\": \"ojas-metal\", \"row\": \"_device\", \"status\": \"info\"}\n{\"runtime\": \"ojas-metal\", \"row\": \"sweep_silu_n1\", \"status\": \"ok\", \"min_ms\": 1.0, \"median_ms\": 1.1}\n{\"runtime\": \"ojas-metal\", \"row\": \"sweep_silu_n1000\", \"status\": \"ok\", \"min_ms\": 1.5, \"median_ms\": 1.6}\n{\"runtime\": \"ojas-metal\", \"row\": \"sweep_silu_n1000000\", \"status\": \"ok\", \"min_ms\": 4.0, \"median_ms\": 4.4}\n{\"runtime\": \"ojas-metal\", \"row\": \"sweep_decode_b1\", \"status\": \"ok\", \"min_ms\": 2.0, \"median_ms\": 2.2}\n{\"runtime\": \"ojas-metal\", \"row\": \"sweep_decode_b4\", \"status\": \"ok\", \"min_ms\": 3.0, \"median_ms\": 3.3}\n{\"runtime\": \"ojas-metal\", \"row\": \"sweep_decode_x4\", \"status\": \"ok\", \"min_ms\": 7.0, \"median_ms\": 7.7}\n{\"runtime\": \"ojas-wgpu\", \"row\": \"sweep_silu_n1000\", \"status\": \"ok\", \"min_ms\": 2.0, \"median_ms\": 2.1}\n{\"runtime\": \"torch-mps\", \"row\": \"sweep_silu_n1\", \"status\": \"ok\", \"min_ms\": 0.5, \"median_ms\": 0.6}\n{\"runtime\": \"torch-mps\", \"row\": \"sweep_silu_n1000\", \"status\": \"failed\", \"detail\": \"x\"}\n{\"runtime\": \"torch-mps\", \"row\": \"sweep_decode_b4\", \"status\": \"ok\", \"min_ms\": 1.0, \"median_ms\": 1.2}\n\n",
	"sweep/round2/m.jsonl":          "{\"runtime\": \"ojas-metal\", \"row\": \"sweep_silu_n1\", \"status\": \"ok\", \"min_ms\": 1.2, \"median_ms\": 1.3}\n{\"runtime\": \"ojas-metal\", \"row\": \"sweep_silu_n1000\", \"status\": \"ok\", \"min_ms\": 1.4, \"median_ms\": 1.9}\n{\"runtime\": \"ojas-metal\", \"row\": \"sweep_silu_n1000000\", \"status\": \"ok\", \"min_ms\": 3.0, \"median_ms\": 5.0}\n{\"runtime\": \"ojas-metal\", \"row\": \"sweep_decode_b4\", \"status\": \"ok\", \"min_ms\": 3.1, \"median_ms\": 3.0}\n{\"runtime\": \"torch-mps\", \"row\": \"sweep_silu_n1000\", \"status\": \"ok\", \"min_ms\": 0.7, \"median_ms\": 0.8}\n",
	"dec/round1/metal-before.jsonl": "{\"runtime\": \"ojas-metal\", \"row\": \"_load\", \"status\": \"info\"}\n{\"runtime\": \"ojas-metal\", \"row\": \"gen_prefill_p32\", \"status\": \"ok\", \"min_ms\": 9.0, \"median_ms\": 9.5}\n{\"runtime\": \"ojas-metal\", \"row\": \"gen_greedy_p32_n32\", \"status\": \"ok\", \"min_ms\": 320.0, \"median_ms\": 330.25}\n",
	"dec/round1/metal-after.jsonl":  "{\"runtime\": \"ojas-metal\", \"row\": \"gen_prefill_p32\", \"status\": \"ok\", \"min_ms\": 8.5, \"median_ms\": 9.0}\n{\"runtime\": \"ojas-metal\", \"row\": \"gen_greedy_p32_n32\", \"status\": \"ok\", \"min_ms\": 300, \"median_ms\": 310}\n",
	"dec/round2/metal-before.jsonl": "{\"runtime\": \"ojas-metal\", \"row\": \"gen_prefill_p32\", \"status\": \"ok\", \"min_ms\": 9.2, \"median_ms\": 9.9}\n{\"runtime\": \"ojas-metal\", \"row\": \"gen_greedy_p32_n32\", \"status\": \"ok\", \"min_ms\": 321.0, \"median_ms\": 331.0}\n",
	"dec/round2/metal-after.jsonl":  "{\"runtime\": \"ojas-metal\", \"row\": \"gen_greedy_p32_n32\", \"status\": \"ok\", \"min_ms\": 299.0, \"median_ms\": 305.5}\n",
}

const wantBenchOps = "rounds: ojas 2, torch 2; times in ms, min over rounds\n| op | dir | shape | ojas 1t | ojas 6t | torch | ratio 1t | ratio 6t | torch alt | spread>15% | parity |\n|---|---|---|---|---|---|---|---|---|---|---|\n| add | fwd | [4,8] | 1.000 | 0.500 | 2.000 | 2.00 | 4.00 | — | ojas1t 30%, torch 50% | pass 1.5e-07 (tol 1e-5) |\n| sdpa | fwd | [1,2,3] | 11.0 | — | 10.0 (math) | 0.91 | — | 9.000 (default) | torch 30% | FAIL (tol 1e-4); FAIL o: ojas off f64 by 1.0e-02 (torch 1.0e-06) |\n| clip | step | [9] | 0.200 | — | 0.000 | — | — | — | ojas1t 25% | FAIL (tol 1e-4); norm: torch off f64 by 4.0e-03, ojas 1.0e-08; g: 4.0e-03, inherits torch's f64 miss above; FAIL h 5.0e-01 |\n| block | fwd | [b] | 20.0 | — | 25.0 (math) | 1.25 | — | — | no | FAIL (tol 1e-3); FAIL y nan |\n| block | step | [b] | 45.0 | — | 50.0 (math) | 1.11 | — | — | no | MISSING |\n| rms_norm | fwd | [r] | 0.100 | — | — | — | — | — | no | MISSING |\n| rms_norm | bwd | [r] | 0.200 | — | — | — | — | — | no | MISSING |\n| add | bwd | [4,8] | 0.300 | — | — | — | — | — | no | MISSING |\nblock, ojas 1t: sum of op rows fwd 13.20 + bwd 6.40 = 19.60 ms; measured fwd 20.0, step 45.0, step - fwd 25.0\nblock, ojas 6t: sum of op rows fwd 1.00 + bwd 3.00 = 4.00 ms; measured fwd —, step —, step - fwd —\nblock, torch: sum of op rows fwd 14.00 + bwd 12.00 = 26.00 ms; measured fwd 25.0, step 50.0, step - fwd 25.0\n"

const wantCPUAB = "rounds=3; cells are ms, min over rounds of each round's min-of-N\n\n| row | dir | base | after | after/base (min) | after/base (median of rounds) | torch | after/torch |\n| :-- | :-- | --: | --: | --: | --: | --: | --: |\n| adamw_50304x768 | step | 8.0000 | 4.0000 | 0.50 | 0.50 | 5.0000 | 0.80 |\n| conv | bwd/im2col |  | 3.0000 |  |  |  |  |\n| mul | fwd | 2.0000 | 0.9000 | 0.45 | 0.49 | 1.5000 | 0.60 |\n| tape | fanin | 5.0000 | 4.0000 | 0.80 | 0.80 |  |  |\n"

const wantSweepFit = "## silu_forward over N values\n\nCells: min-of-round-minimums median (m) / median-of-medians (d) ms, spread of per-round medians.\n\n| N | ojas-metal | ojas-wgpu | torch-mps | metal/torch d ratio |\n| --: | :-- | :-- | :-- | --: |\n| 1 | 1.100 / 1.200 (18%) | missing | 0.500 / 0.600 (0%) | 0.50x |\n| 1000 | 1.450 / 1.750 (19%) | 2.000 / 2.100 (0%) | failed | -x |\n| 1000000 | 3.500 / 4.700 (14%) | missing | missing | -x |\n\n## Crossover where the per-value work equals the fixed cost\n\n| runtime | stat | floor a = t(1) ms | slope b ns/value | model N* = a/b | direct N (t = 2a) |\n| :-- | :-- | --: | --: | --: | --: |\n| ojas-metal | m | 1.100 | 2.0521 | 0.54M | 0.37M |\n| ojas-metal | d | 1.200 | 2.9530 | 0.41M | 0.22M |\n| ojas-wgpu | m | incomplete (1 sizes) | | | |\n| ojas-wgpu | d | incomplete (1 sizes) | | | |\n| torch-mps | m | incomplete (1 sizes) | | | |\n| torch-mps | d | incomplete (1 sizes) | | | |\n\n## Decode: B requests batched (b) vs B batch-1 calls, one sync (x)\n\nCells: median-of-medians ms (spread). Per-request = cell / B. Extra per call = (x - b) / (B - 1).\n\n| B | runtime | batched b | per request | split x | per request | extra per call |\n| --: | :-- | --: | --: | --: | --: | --: |\n| 1 | ojas-metal | 2.200 (0%) | 2.200 | - | - | - |\n| 1 | ojas-wgpu | missing | - | - | - | - |\n| 1 | torch-mps | missing | - | - | - | - |\n| 4 | ojas-metal | 3.150 (10%) | 0.787 | 7.700 (0%) | 1.925 | 1.517 |\n| 4 | ojas-wgpu | missing | - | missing | - | - |\n| 4 | torch-mps | 1.200 (0%) | 0.300 | missing | - | - |\n"

const wantDecodeAB = "| lane | row | before min_ms | after min_ms | after / before per round, median [min-max] | rounds |\n| :-- | :-- | --: | --: | :-- | --: |\n| metal | gen_greedy_p32_n32 | 320.50 | 299.50 | 0.934 [0.931-0.938] | 2 |\n| metal | gen_prefill_p32 | 9.00 | 8.50 | 0.944 [0.944-0.944] | 1 |\n\n| lane | decode row | before ms / token | after ms / token | after - before ms / token, per round median [min-max] |\n| :-- | :-- | --: | --: | :-- |\n| metal | gen_greedy_p32_n32 | 10.045 | 9.403 | -0.629 [-0.629--0.629] |\n"

// Python 3.14: sum([0.1] * 10) and sum([1e16, 1.0, -1e16]).
const wantSumTenths = 1.0
const wantSumCancel = 1.0
