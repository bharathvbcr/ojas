// sitegen renders the repository's markdown docs and crate READMEs into the
// static reference library under site/reference/.
//
// The markdown files are the source of truth, and site/ is the only generated
// tree: nothing is mirrored into docs/. Run it from the repository root:
//
//	go run scripts/sitegen/main.go          # write site/reference/ and the test counts
//	go run scripts/sitegen/main.go -check   # exit 1 if anything generated is stale
//
// Preview locally by serving site/ (for example python3 -m http.server -d site).
//
// Test counts come from one recorded file, docs/data/test-runs.json. Marked
// spans in README.md, docs/status.md, docs/backends.md and site/index.html
// (see countRe) are rewritten from it, so a number is never typed twice.
//
// Tests: go test scripts/sitegen/main.go scripts/sitegen/main_test.go
//
// It uses only the standard library.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"html"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	siteURL = "https://ojas.vbcr.dev"
	ghBlob  = "https://github.com/bharathvbcr/ojas/blob/main/"
	ghTree  = "https://github.com/bharathvbcr/ojas/tree/main/"
	ghRaw   = "https://raw.githubusercontent.com/bharathvbcr/ojas/main/"
	outDir  = "site/reference"
)

type category struct {
	Key, Title, Blurb string
}

var categories = []category{
	{"design", "Design and contracts", "How the engine is put together and the rules every backend follows."},
	{"status", "Status and plans", "What passes today, what is planned, and the limits that remain."},
	{"bench", "Benchmarks", "Raw tables and commands behind the numbers on the home page."},
	{"audit", "Audits", "The defects found in the author's own engines and how each was closed."},
	{"crates", "Crates", "One README per workspace crate, plus the project README."},
}

// docCategory maps docs/<slug>.md to a category. A doc missing from the map is
// an error, so a new file cannot be published uncategorised by accident.
var docCategory = map[string]string{
	"architecture": "design", "framework-design": "design", "shape-contract": "design",
	"dtype-policy": "design", "op-coverage": "design", "backends": "design",
	"checkpoint-v1": "design", "adaptive-resources": "design", "data-formats": "design",
	"status": "status", "status-archive-2026-10-05": "status", "baseline": "status", "typed-storage-plan": "status",
	"pytorch-parity-plan": "status", "cuda-backend-scoping": "status", "metal-deferred-faults": "status",
	"bench-cpu-vs-torch": "bench", "bench-gpu-vs-torch": "bench", "bench-plots": "bench",
	"audit": "audit", "audit-phase2": "audit", "audit-phase3": "audit", "audit-cpu-hot": "audit",
	"audit-larger-step": "audit", "audit-resources": "audit", "audit-close": "audit",
}

// docSkip lists docs that describe the site itself and are not part of the library.
var docSkip = map[string]bool{"web-and-domain": true}

type doc struct {
	Slug, Src, Cat, Title, Desc string
	Body                        string
	Heads                       []head
}

type head struct {
	Level int
	ID    string
	Text  string
}

func main() {
	check := flag.Bool("check", false, "verify every generated file (reference pages, test counts) is current instead of writing it")
	flag.Parse()
	if err := run(*check); err != nil {
		fmt.Fprintln(os.Stderr, "sitegen:", err)
		os.Exit(1)
	}
}

func run(check bool) error {
	if _, err := os.Stat("Cargo.toml"); err != nil {
		return fmt.Errorf("run from the repository root: %w", err)
	}
	tracked, err := loadTracked()
	if err != nil {
		return err
	}
	members, err := workspaceMembersFile("Cargo.toml")
	if err != nil {
		return err
	}
	runs, err := loadTestRuns(testRunsPath)
	if err != nil {
		return err
	}
	docs, err := collect(tracked, members)
	if err != nil {
		return err
	}
	bySrc := map[string]*doc{}
	for _, d := range docs {
		bySrc[d.Src] = d
	}
	files := map[string][]byte{}
	// Count-bearing sources are regenerated first; the pages render the
	// regenerated text, so a page can never show a number its source does not.
	sources := map[string]string{}
	for _, p := range countFiles {
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		esc := func(v string) string { return v }
		if strings.HasSuffix(p, ".html") {
			esc = html.EscapeString
		}
		gen, n, err := applyCountsWith(string(raw), runs, esc)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		if n == 0 {
			return fmt.Errorf("%s has no count markers; wrap its numbers in <!--count:KEY-->...<!--/count-->", p)
		}
		sources[p] = gen
		files[p] = []byte(gen)
	}
	unresolved := 0
	for _, d := range docs {
		text, ok := sources[d.Src]
		if !ok {
			src, err := os.ReadFile(d.Src)
			if err != nil {
				return err
			}
			text = string(src)
		}
		r := &renderer{tracked: tracked, doc: d, bySrc: bySrc, ids: map[string]int{}}
		lines := splitLines(text)
		r.scanRaw(lines)
		d.Body = r.blocks(lines)
		d.Heads = r.heads
		d.Title = r.title
		if d.Title == "" {
			d.Title = d.Slug
		}
		d.Desc = r.desc
		unresolved += r.unresolved
	}
	if unresolved > 0 {
		return fmt.Errorf("%d links in the markdown point at nothing; fix the source", unresolved)
	}
	for i, d := range docs {
		var prev, next *doc
		if i > 0 {
			prev = docs[i-1]
		}
		if i+1 < len(docs) {
			next = docs[i+1]
		}
		files[outDir+"/"+d.Slug+".html"] = []byte(pageHTML(d, docs, prev, next))
	}
	files[outDir+"/index.html"] = []byte(indexHTML(docs))
	files["site/assets/reference-index.json"] = searchJSON(docs)
	files["site/sitemap.xml"] = []byte(sitemap(docs))

	if check {
		return verify(files)
	}
	n := 0
	for name, b := range files {
		if got, err := os.ReadFile(name); err == nil && bytes.Equal(got, b) {
			continue
		}
		if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(name, b, 0o644); err != nil {
			return err
		}
		n++
	}
	for _, f := range orphans(files) {
		if err := os.Remove(f); err != nil {
			return err
		}
	}
	fmt.Printf("sitegen: %d of %d generated files changed (%d reference pages in site/reference)\n", n, len(files), len(docs)+1)
	return nil
}

// orphans lists generated pages on disk that no longer have a source.
func orphans(files map[string][]byte) []string {
	var out []string
	existing, _ := filepath.Glob(filepath.Join(outDir, "*.html"))
	for _, f := range existing {
		if _, ok := files[filepath.ToSlash(f)]; !ok {
			out = append(out, f)
		}
	}
	return out
}

func verify(files map[string][]byte) error {
	var bad []string
	for name, b := range files {
		got, err := os.ReadFile(name)
		if err != nil || !bytes.Equal(got, b) {
			bad = append(bad, name)
		}
	}
	for _, f := range orphans(files) {
		bad = append(bad, f+" (orphan)")
	}
	if len(bad) > 0 {
		sort.Strings(bad)
		return fmt.Errorf("generated files are stale (%d): %s\nrun: go run scripts/sitegen/main.go", len(bad), strings.Join(bad, ", "))
	}
	fmt.Printf("sitegen: generated files are up to date (%d files)\n", len(files))
	return nil
}

func searchJSON(docs []*doc) []byte {
	var entries []searchEntry
	for _, d := range docs {
		e := searchEntry{Slug: d.Slug, Title: navLabel(d), Cat: catTitle(d.Cat), Desc: d.Desc}
		for _, h := range d.Heads {
			e.Heads = append(e.Heads, h.Text)
		}
		entries = append(entries, e)
	}
	b, _ := json.Marshal(entries)
	return b
}

func sitemap(docs []*doc) string {
	var b strings.Builder
	b.WriteString("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n")
	b.WriteString("  <url><loc>" + siteURL + "/</loc></url>\n")
	b.WriteString("  <url><loc>" + siteURL + "/reference/</loc></url>\n")
	for _, d := range docs {
		b.WriteString("  <url><loc>" + siteURL + "/reference/" + d.Slug + ".html</loc></url>\n")
	}
	b.WriteString("</urlset>\n")
	return b.String()
}

// collect lists every tracked source file in a stable order: category, then
// name. Untracked files are ignored so a local scratch directory cannot add a
// page that a clean checkout (CI) would not have. Crate pages exist only for
// workspace members; other directories with a README (bench, go, gemma-metal)
// are reached through GitHub links.
func collect(tracked trackedSet, members []string) ([]*doc, error) {
	var out []*doc
	mds, err := filepath.Glob("docs/*.md")
	if err != nil {
		return nil, err
	}
	for _, p := range mds {
		if _, ok := tracked[filepath.ToSlash(p)]; !ok {
			continue
		}
		slug := strings.TrimSuffix(filepath.Base(p), ".md")
		if docSkip[slug] {
			continue
		}
		cat, ok := docCategory[slug]
		if !ok {
			return nil, fmt.Errorf("%s has no category: add it to docCategory in scripts/sitegen/main.go", p)
		}
		out = append(out, &doc{Slug: slug, Src: filepath.ToSlash(p), Cat: cat})
	}
	for _, dir := range members {
		p := dir + "/README.md"
		if _, ok := tracked[p]; !ok {
			continue
		}
		out = append(out, &doc{Slug: "crate-" + strings.ReplaceAll(dir, "/", "-"), Src: p, Cat: "crates"})
	}
	out = append(out, &doc{Slug: "readme", Src: "README.md", Cat: "crates"})
	order := map[string]int{}
	for i, c := range categories {
		order[c.Key] = i
	}
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Cat != b.Cat {
			return order[a.Cat] < order[b.Cat]
		}
		if (a.Slug == "readme") != (b.Slug == "readme") {
			return a.Slug == "readme"
		}
		return a.Slug < b.Slug
	})
	return out, nil
}

// tomlStrings reads the string array assigned to key in the [workspace] table.
// It handles comments (including brackets inside them), both quote styles and
// multi-line arrays; a key that is absent returns nil.
func tomlStrings(doc, key string) ([]string, error) {
	inWorkspace := false
	lines := strings.Split(doc, "\n")
	for i := 0; i < len(lines); i++ {
		line := strings.TrimSpace(lines[i])
		if strings.HasPrefix(line, "[") && !strings.HasPrefix(line, "[[") {
			end := strings.Index(line, "]")
			inWorkspace = end > 0 && strings.TrimSpace(line[1:end]) == "workspace"
			continue
		}
		if !inWorkspace || !strings.HasPrefix(line, key) {
			continue
		}
		rest := strings.TrimSpace(line[len(key):])
		if !strings.HasPrefix(rest, "=") {
			continue
		}
		// Scan from the opening bracket, across lines, until its closing one.
		text := strings.TrimSpace(rest[1:])
		for j := i + 1; ; j++ {
			vals, closed, err := scanTOMLArray(text)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", key, err)
			}
			if closed {
				return vals, nil
			}
			if j >= len(lines) {
				return nil, fmt.Errorf("%s: unterminated array", key)
			}
			text += "\n" + lines[j]
		}
	}
	return nil, nil
}

// scanTOMLArray parses text that starts at '['. closed is false when the array
// has not ended yet.
func scanTOMLArray(text string) (vals []string, closed bool, err error) {
	if !strings.HasPrefix(text, "[") {
		return nil, false, fmt.Errorf("expected an array")
	}
	for i := 1; i < len(text); i++ {
		switch c := text[i]; c {
		case '#':
			for i < len(text) && text[i] != '\n' {
				i++
			}
		case '"':
			var b strings.Builder
			for i++; i < len(text) && text[i] != '"'; i++ {
				if text[i] == '\\' && i+1 < len(text) {
					i++
				}
				b.WriteByte(text[i])
			}
			if i >= len(text) {
				return nil, false, nil
			}
			vals = append(vals, b.String())
		case '\'':
			j := strings.IndexByte(text[i+1:], '\'')
			if j < 0 {
				return nil, false, nil
			}
			vals = append(vals, text[i+1:i+1+j])
			i += j + 1
		case ']':
			return vals, true, nil
		}
	}
	return nil, false, nil
}

// workspaceMembers reads the `members` array of a workspace Cargo.toml (patterns
// as written; see expandMembers for globs).
func workspaceMembers(cargoToml string) ([]string, error) {
	out, err := tomlStrings(cargoToml, "members")
	if err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no workspace members array")
	}
	return out, nil
}

// expandMembers expands glob patterns against the filesystem and drops the
// workspace `exclude` entries. A pattern that matches nothing is an error.
func expandMembers(patterns, exclude []string) ([]string, error) {
	skip := map[string]bool{}
	for _, e := range exclude {
		skip[path.Clean(e)] = true
	}
	var out []string
	for _, pat := range patterns {
		matches := []string{path.Clean(pat)}
		if strings.ContainsAny(pat, "*?[") {
			g, err := filepath.Glob(pat)
			if err != nil {
				return nil, fmt.Errorf("member pattern %q: %w", pat, err)
			}
			matches = matches[:0]
			for _, m := range g {
				if fi, err := os.Stat(filepath.Join(m, "Cargo.toml")); err == nil && !fi.IsDir() {
					matches = append(matches, filepath.ToSlash(m))
				}
			}
			if len(matches) == 0 {
				return nil, fmt.Errorf("member pattern %q matches no crate directory", pat)
			}
			sort.Strings(matches)
		}
		for _, m := range matches {
			if !skip[m] {
				out = append(out, m)
			}
		}
	}
	return out, nil
}

func workspaceMembersFile(p string) ([]string, error) {
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	pats, err := workspaceMembers(string(b))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", p, err)
	}
	excl, err := tomlStrings(string(b), "exclude")
	if err != nil {
		return nil, fmt.Errorf("%s: %w", p, err)
	}
	return expandMembers(pats, excl)
}

func splitLines(s string) []string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	return strings.Split(strings.TrimRight(s, "\n"), "\n")
}

// ---------------------------------------------------------------- markdown

// trackedSet is what a fresh checkout contains: files from `git ls-files` and
// their parent directories. Links are checked against it rather than the local
// disk, so ignored build output cannot make a link look valid here and dead in CI.
type trackedSet map[string]bool // path -> is a directory

func loadTracked() (trackedSet, error) {
	out, err := exec.Command("git", "ls-files", "-z").Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-files (needed to validate links against a clean checkout): %w", err)
	}
	t := trackedSet{}
	for _, f := range strings.Split(string(out), "\x00") {
		if f == "" {
			continue
		}
		t[f] = false
		for d := path.Dir(f); d != "."; d = path.Dir(d) {
			t[d] = true
		}
	}
	return t, nil
}

func (t trackedSet) lookup(p string) (isDir, ok bool) {
	isDir, ok = t[p]
	return
}

type renderer struct {
	tracked    trackedSet
	doc        *doc
	bySrc      map[string]*doc
	ids        map[string]int
	heads      []head
	title      string
	desc       string
	unresolved int
}

var (
	reHTMLBlock   = regexp.MustCompile(`^</?(p|div|img|details|summary|br|table|center)[\s>/]`)
	reHeading     = regexp.MustCompile(`^(#{1,6})\s+(.*?)\s*#*\s*$`)
	reFence       = regexp.MustCompile("^(\\s*)(```+|~~~+)\\s*([A-Za-z0-9_+-]*)")
	reUL          = regexp.MustCompile(`^(\s*)([-*+])\s+(.*)$`)
	reOL          = regexp.MustCompile(`^(\s*)(\d+)[.)]\s+(.*)$`)
	reHR          = regexp.MustCompile(`^\s{0,3}(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$`)
	reTableSp     = regexp.MustCompile(`^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$`)
	reCommentLine = regexp.MustCompile(`^\s*(<!--.*?-->\s*)+$`)
	reAlert       = regexp.MustCompile(`^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$`)
)

func (r *renderer) slug(text string) string {
	s := strings.ToLower(stripInline(text))
	var b strings.Builder
	for _, c := range s {
		switch {
		case c >= 'a' && c <= 'z', c >= '0' && c <= '9':
			b.WriteRune(c)
		case c == ' ' || c == '-' || c == '_':
			b.WriteByte('-')
		}
	}
	id := strings.Trim(b.String(), "-")
	if id == "" {
		id = "section"
	}
	n := r.ids[id]
	r.ids[id] = n + 1
	if n > 0 {
		id = fmt.Sprintf("%s-%d", id, n)
	}
	return id
}

func stripInline(s string) string {
	s = regexp.MustCompile("`([^`]*)`").ReplaceAllString(s, "$1")
	s = regexp.MustCompile(`\[([^\]]*)\]\([^)]*\)`).ReplaceAllString(s, "$1")
	s = strings.NewReplacer("**", "", "__", "", "*", "").Replace(s)
	return strings.TrimSpace(s)
}

func isBlockStart(l string) bool {
	return reHeading.MatchString(l) || reFence.MatchString(l) || reHR.MatchString(l) ||
		reUL.MatchString(l) || reOL.MatchString(l) || strings.HasPrefix(strings.TrimSpace(l), ">") ||
		strings.HasPrefix(strings.TrimSpace(l), "|")
}

func indentOf(l string) int {
	n := 0
	for _, c := range l {
		if c == ' ' {
			n++
		} else if c == '\t' {
			n += 4
		} else {
			break
		}
	}
	return n
}

func dedent(l string, n int) string {
	i := 0
	for i < len(l) && n > 0 && (l[i] == ' ' || l[i] == '\t') {
		if l[i] == '\t' {
			n -= 4
		} else {
			n--
		}
		i++
	}
	return l[i:]
}

func (r *renderer) blocks(lines []string) string {
	var out strings.Builder
	i := 0
	for i < len(lines) {
		l := lines[i]
		if strings.TrimSpace(l) == "" {
			i++
			continue
		}
		if reCommentLine.MatchString(l) {
			i++
			continue
		}
		if reHTMLBlock.MatchString(l) {
			// A raw HTML block (the README's centred logo): not markdown, so skip it.
			for i < len(lines) && strings.TrimSpace(lines[i]) != "" {
				i++
			}
			continue
		}
		if m := reFence.FindStringSubmatch(l); m != nil {
			ind, mark, lang := indentOf(m[1]), m[2], m[3]
			var code []string
			i++
			for i < len(lines) && !strings.HasPrefix(strings.TrimSpace(lines[i]), mark[:3]) {
				code = append(code, dedent(lines[i], ind))
				i++
			}
			i++
			out.WriteString(codeBlock(lang, strings.Join(code, "\n")))
			continue
		}
		if m := reHeading.FindStringSubmatch(l); m != nil {
			lvl := len(m[1])
			id := r.slug(m[2])
			if lvl == 1 && r.title == "" {
				r.title = stripInline(m[2])
			} else if lvl == 2 || lvl == 3 {
				r.heads = append(r.heads, head{lvl, id, stripInline(m[2])})
			}
			fmt.Fprintf(&out, "<h%d id=\"%s\">%s <a class=\"anchor\" href=\"#%s\" aria-label=\"Link to this section\">#</a></h%d>\n",
				lvl, id, r.inline(m[2]), id, lvl)
			i++
			continue
		}
		if reHR.MatchString(l) && !reUL.MatchString(l) {
			out.WriteString("<hr>\n")
			i++
			continue
		}
		if strings.HasPrefix(strings.TrimSpace(l), "|") && i+1 < len(lines) && reTableSp.MatchString(lines[i+1]) {
			var rows []string
			for i < len(lines) && strings.HasPrefix(strings.TrimSpace(lines[i]), "|") {
				rows = append(rows, lines[i])
				i++
			}
			out.WriteString(r.table(rows))
			continue
		}
		if strings.HasPrefix(strings.TrimSpace(l), ">") {
			var inner []string
			for i < len(lines) && strings.HasPrefix(strings.TrimSpace(lines[i]), ">") {
				t := strings.TrimPrefix(strings.TrimSpace(lines[i]), ">")
				inner = append(inner, strings.TrimPrefix(t, " "))
				i++
			}
			out.WriteString(r.quote(inner))
			continue
		}
		if reUL.MatchString(l) || reOL.MatchString(l) {
			var s string
			s, i = r.list(lines, i)
			out.WriteString(s)
			continue
		}
		var para []string
		for i < len(lines) && strings.TrimSpace(lines[i]) != "" && (len(para) == 0 || !isBlockStart(lines[i])) {
			para = append(para, strings.TrimSpace(lines[i]))
			i++
		}
		text := strings.Join(para, " ")
		if r.desc == "" && r.title != "" && len(text) > 40 {
			r.desc = clip(stripInline(text), 200)
		}
		fmt.Fprintf(&out, "<p>%s</p>\n", r.inline(text))
	}
	return out.String()
}

func clip(s string, n int) string {
	rs := []rune(s)
	if len(rs) <= n {
		return s
	}
	return strings.TrimSpace(string(rs[:n-1])) + "…"
}

func (r *renderer) quote(inner []string) string {
	if len(inner) > 0 {
		if m := reAlert.FindStringSubmatch(strings.TrimSpace(inner[0])); m != nil {
			kind := strings.ToLower(m[1])
			return fmt.Sprintf("<aside class=\"alert alert-%s\"><p class=\"alert-title\">%s</p>%s</aside>\n",
				kind, strings.Title(kind), r.blocks(inner[1:]))
		}
	}
	return "<blockquote>" + r.blocks(inner) + "</blockquote>\n"
}

// list renders a list starting at lines[i] and returns the index after it.
func (r *renderer) list(lines []string, i int) (string, int) {
	first := lines[i]
	base := indentOf(first)
	ordered := reOL.MatchString(first)
	tag := "ul"
	if ordered {
		tag = "ol"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "<%s>\n", tag)
	for i < len(lines) {
		l := lines[i]
		var m []string
		if ordered {
			m = reOL.FindStringSubmatch(l)
		} else {
			m = reUL.FindStringSubmatch(l)
		}
		if m == nil || indentOf(l) != base {
			break
		}
		content := []string{m[3]}
		i++
		for i < len(lines) {
			n := lines[i]
			if strings.TrimSpace(n) == "" {
				// A blank line continues the item only if indented content follows.
				j := i
				for j < len(lines) && strings.TrimSpace(lines[j]) == "" {
					j++
				}
				if j < len(lines) && indentOf(lines[j]) > base {
					for k := i; k < j; k++ {
						content = append(content, "")
					}
					i = j
					continue
				}
				break
			}
			if indentOf(n) > base {
				content = append(content, dedent(n, base+2))
				i++
				continue
			}
			if indentOf(n) == base && (reUL.MatchString(n) || reOL.MatchString(n)) {
				break
			}
			if indentOf(n) <= base && isBlockStart(n) {
				break
			}
			content = append(content, strings.TrimSpace(n)) // lazy continuation
			i++
		}
		b.WriteString("<li>" + r.item(content) + "</li>\n")
	}
	fmt.Fprintf(&b, "</%s>\n", tag)
	return b.String(), i
}

// item renders a list item: a lone paragraph stays inline (tight list).
func (r *renderer) item(content []string) string {
	inner := r.blocks(content)
	if strings.Count(inner, "<p>") == 1 && strings.HasPrefix(inner, "<p>") {
		end := strings.Index(inner, "</p>")
		if end >= 0 {
			return inner[3:end] + inner[end+4:]
		}
	}
	return inner
}

func splitRow(row string) []string {
	row = strings.TrimSpace(row)
	row = strings.TrimPrefix(row, "|")
	row = strings.TrimSuffix(row, "|")
	var cells []string
	var cur strings.Builder
	inCode := false
	for i := 0; i < len(row); i++ {
		c := row[i]
		switch {
		case c == '`':
			inCode = !inCode
			cur.WriteByte(c)
		case c == '\\' && i+1 < len(row) && row[i+1] == '|':
			cur.WriteByte('|')
			i++
		case c == '|' && !inCode:
			cells = append(cells, strings.TrimSpace(cur.String()))
			cur.Reset()
		default:
			cur.WriteByte(c)
		}
	}
	return append(cells, strings.TrimSpace(cur.String()))
}

func (r *renderer) table(rows []string) string {
	head := splitRow(rows[0])
	aligns := splitRow(rows[1])
	al := make([]string, len(head))
	for i := range head {
		if i < len(aligns) {
			a := aligns[i]
			switch {
			case strings.HasPrefix(a, ":") && strings.HasSuffix(a, ":"):
				al[i] = "center"
			case strings.HasSuffix(a, ":"):
				al[i] = "right"
			}
		}
	}
	cell := func(tag string, i int, s string) string {
		a := ""
		if i < len(al) && al[i] != "" {
			a = ` style="text-align:` + al[i] + `"`
		}
		return fmt.Sprintf("<%s%s>%s</%s>", tag, a, r.inline(s), tag)
	}
	var b strings.Builder
	b.WriteString("<div class=\"table-wrap\"><table>\n<thead><tr>")
	for i, h := range head {
		b.WriteString(cell("th", i, h))
	}
	b.WriteString("</tr></thead>\n<tbody>\n")
	for _, row := range rows[2:] {
		b.WriteString("<tr>")
		cells := splitRow(row)
		for i := range head {
			s := ""
			if i < len(cells) {
				s = cells[i]
			}
			b.WriteString(cell("td", i, s))
		}
		b.WriteString("</tr>\n")
	}
	b.WriteString("</tbody></table></div>\n")
	return b.String()
}

func codeBlock(lang, code string) string {
	esc := html.EscapeString(code)
	if lang == "mermaid" {
		// reference.js loads Mermaid from cdnjs and replaces this element with the
		// drawn SVG; if the script cannot load, the source stays readable.
		return "<figure class=\"diagram\"><pre class=\"mermaid\">" + esc + "</pre></figure>\n"
	}
	label := ""
	if lang != "" {
		label = fmt.Sprintf("<span class=\"code-lang\">%s</span>", html.EscapeString(lang))
	}
	return fmt.Sprintf("<div class=\"code\">%s<button type=\"button\" class=\"copy\" aria-label=\"Copy code\">Copy</button><pre><code>%s</code></pre></div>\n", label, esc)
}

var (
	reCode    = regexp.MustCompile("`+([^`]+?)`+")
	reLink    = regexp.MustCompile(`\[((?:[^\[\]]|\[[^\[\]]*\])+)\]\((?:<([^>]*)>|([^)\s]+))(?:\s+"[^"]*")?\)`)
	reImage   = regexp.MustCompile(`!\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\((?:<([^>]*)>|([^)\s]+))(?:\s+"([^"]*)")?\)`)
	reComment = regexp.MustCompile(`(?s)<!--.*?-->`)
	reBold    = regexp.MustCompile(`\*\*(.+?)\*\*|__(.+?)__`)
	reItalic  = regexp.MustCompile(`(^|[^\w*])\*([^*\s][^*]*?)\*([^\w*]|$)`)
	reAuto    = regexp.MustCompile(`(^|[\s(])(https?://[^\s<)]+[^\s<).,;:])`)
)

func (r *renderer) inline(s string) string {
	var codes []string
	s = reCode.ReplaceAllStringFunc(s, func(m string) string {
		inner := reCode.FindStringSubmatch(m)[1]
		codes = append(codes, "<code>"+html.EscapeString(strings.TrimSpace(inner))+"</code>")
		return fmt.Sprintf("\x00%d\x00", len(codes)-1)
	})
	s = reComment.ReplaceAllString(s, "")
	// Images first, so [![alt](img)](link) keeps its image as the link label.
	var images []string
	s = reImage.ReplaceAllStringFunc(s, func(m string) string {
		sm := reImage.FindStringSubmatch(m)
		images = append(images, r.image(sm[1], sm[2]+sm[3], sm[4]))
		return fmt.Sprintf("\x02%d\x02", len(images)-1)
	})
	var links []string
	s = reLink.ReplaceAllStringFunc(s, func(m string) string {
		sm := reLink.FindStringSubmatch(m)
		links = append(links, r.link(sm[1], sm[2]+sm[3]))
		return fmt.Sprintf("\x01%d\x01", len(links)-1)
	})
	s = html.EscapeString(s)
	s = reAuto.ReplaceAllString(s, `$1<a href="$2" rel="noopener">$2</a>`)
	s = reBold.ReplaceAllString(s, "<strong>$1$2</strong>")
	s = reItalic.ReplaceAllString(s, "$1<em>$2</em>$3")
	for i, l := range links {
		s = strings.Replace(s, fmt.Sprintf("\x01%d\x01", i), l, 1)
	}
	for i, im := range images {
		s = strings.Replace(s, fmt.Sprintf("\x02%d\x02", i), im, 1)
	}
	for i, c := range codes {
		s = strings.Replace(s, fmt.Sprintf("\x00%d\x00", i), c, 1)
	}
	return s
}

// link rewrites a markdown link target to something that works on the site.
func (r *renderer) link(text, target string) string {
	label := r.inlineLabel(text)
	if strings.HasPrefix(target, "http://") || strings.HasPrefix(target, "https://") || strings.HasPrefix(target, "mailto:") {
		return fmt.Sprintf("<a href=\"%s\" rel=\"noopener\">%s</a>", html.EscapeString(target), label)
	}
	if strings.HasPrefix(target, "#") {
		return fmt.Sprintf("<a href=\"%s\">%s</a>", html.EscapeString(target), label)
	}
	frag := ""
	if k := strings.IndexAny(target, "#?"); k >= 0 {
		target, frag = target[:k], target[k:]
	}
	repo, ok := r.repoPath(target)
	if !ok {
		return "<code>" + html.EscapeString(text) + "</code>"
	}
	repo = strings.TrimSuffix(repo, "/")
	if d, ok := r.bySrc[repo]; ok {
		return fmt.Sprintf("<a href=\"%s.html%s\">%s</a>", d.Slug, html.EscapeString(frag), label)
	}
	if repo == "site/index.html" || repo == "docs/index.html" {
		return fmt.Sprintf("<a href=\"../\">%s</a>", label)
	}
	isDir, ok := r.tracked.lookup(repo)
	if !ok {
		fmt.Fprintf(os.Stderr, "sitegen: %s links to %q, which is not tracked by git\n", r.doc.Src, repo)
		r.unresolved++
		return fmt.Sprintf("<code>%s</code>", html.EscapeString(text))
	}
	if isDir {
		if d, ok := r.bySrc[repo+"/README.md"]; ok {
			return fmt.Sprintf("<a href=\"%s.html\">%s</a>", d.Slug, label)
		}
		return fmt.Sprintf("<a href=\"%s%s\" rel=\"noopener\">%s</a>", ghTree, html.EscapeString(escPath(repo)), label)
	}
	return fmt.Sprintf("<a href=\"%s%s%s\" rel=\"noopener\">%s</a>", ghBlob, html.EscapeString(escPath(repo)), html.EscapeString(frag), label)
}

var pathEscaper = strings.NewReplacer("%", "%25", " ", "%20", "?", "%3F", "#", "%23", "\"", "%22", "<", "%3C", ">", "%3E")

// escPath makes a repository path safe inside a URL without touching characters
// (unicode, slashes) that browsers accept.
func escPath(p string) string { return pathEscaper.Replace(p) }

var (
	reRefDef   = regexp.MustCompile(`^\s{0,3}\[[^\]]+\]:\s*<?([^\s>]+)`)
	reAutolink = regexp.MustCompile(`<([A-Za-z][A-Za-z0-9+.-]*:[^>\s]*)>`)
	reHTMLRef  = regexp.MustCompile(`(?i)\b(?:href|src)\s*=\s*["']?([^"'\s>]+)`)
	reCodeSpan = regexp.MustCompile("`+[^`]+`+")
)

// machineLocal reports whether a link target only works on one machine.
func machineLocal(t string) bool {
	l := strings.ToLower(t)
	return strings.HasPrefix(l, "file:") || (strings.HasPrefix(t, "/") && !strings.HasPrefix(t, "//")) || strings.Contains(t, "\\")
}

// scanRaw rejects machine-local targets written in the forms the inline link
// renderer never sees: reference definitions, autolinks and raw HTML href/src.
// Fenced code and code spans are skipped.
func (r *renderer) scanRaw(lines []string) {
	fenced, mark := false, ""
	for _, line := range lines {
		if m := reFence.FindStringSubmatch(line); m != nil {
			if !fenced {
				fenced, mark = true, m[2][:3]
			} else if strings.HasPrefix(strings.TrimSpace(line), mark) {
				fenced = false
			}
			continue
		}
		if fenced {
			continue
		}
		line = reCodeSpan.ReplaceAllString(line, "")
		var targets []string
		if m := reRefDef.FindStringSubmatch(line); m != nil {
			targets = append(targets, m[1])
		}
		for _, m := range reAutolink.FindAllStringSubmatch(line, -1) {
			if strings.HasPrefix(strings.ToLower(m[1]), "file:") {
				targets = append(targets, m[1])
			}
		}
		for _, m := range reHTMLRef.FindAllStringSubmatch(line, -1) {
			targets = append(targets, m[1])
		}
		for _, t := range targets {
			if machineLocal(t) {
				fmt.Fprintf(os.Stderr, "sitegen: %s has a machine-local target %q in a reference, autolink or HTML attribute; use a repo-relative path\n", r.doc.Src, t)
				r.unresolved++
			}
		}
	}
}

func (r *renderer) inlineLabel(text string) string { return r.inline(text) }

// repoPath resolves a relative link target to a repository path. Targets that
// only work on one machine are rejected, never rewritten: file:// URLs, absolute
// paths and backslash paths. ok is false after counting the rejection.
func (r *renderer) repoPath(target string) (repo string, ok bool) {
	switch {
	case strings.HasPrefix(strings.ToLower(target), "file:"):
		fmt.Fprintf(os.Stderr, "sitegen: %s has a file:// link %q; use a repo-relative path\n", r.doc.Src, target)
	case strings.HasPrefix(target, "/"):
		fmt.Fprintf(os.Stderr, "sitegen: %s has an absolute link %q; use a repo-relative path\n", r.doc.Src, target)
	case strings.Contains(target, "\\"):
		fmt.Fprintf(os.Stderr, "sitegen: %s has a backslash in link %q; use forward slashes\n", r.doc.Src, target)
	default:
		dec, err := url.PathUnescape(target)
		if err != nil {
			fmt.Fprintf(os.Stderr, "sitegen: %s has a link %q with a bad percent-escape: %v\n", r.doc.Src, target, err)
			break
		}
		repo := path.Clean(path.Join(path.Dir(r.doc.Src), dec))
		if repo == ".." || strings.HasPrefix(repo, "../") {
			fmt.Fprintf(os.Stderr, "sitegen: %s links to %q, which leaves the repository; name it in code instead of linking it\n", r.doc.Src, target)
			break
		}
		return repo, true
	}
	r.unresolved++
	return "", false
}

// image renders ![alt](src "title"). A tracked file under site/ is served by the
// site itself; any other tracked file is loaded from GitHub raw, so the page never
// points at a path the deployed site does not have.
func (r *renderer) image(alt, target, title string) string {
	attrs := fmt.Sprintf(" alt=\"%s\"", html.EscapeString(stripInline(alt)))
	if title != "" {
		attrs += fmt.Sprintf(" title=\"%s\"", html.EscapeString(title))
	}
	if strings.HasPrefix(target, "http://") || strings.HasPrefix(target, "https://") {
		return fmt.Sprintf("<img src=\"%s\"%s loading=\"lazy\">", html.EscapeString(target), attrs)
	}
	frag := ""
	if k := strings.IndexAny(target, "#?"); k >= 0 {
		target, frag = target[:k], target[k:]
	}
	repo, ok := r.repoPath(target)
	if !ok {
		return "<code>" + html.EscapeString(alt) + "</code>"
	}
	if isDir, tracked := r.tracked.lookup(repo); !tracked || isDir {
		fmt.Fprintf(os.Stderr, "sitegen: %s embeds image %q, which is not a tracked file\n", r.doc.Src, repo)
		r.unresolved++
		return "<code>" + html.EscapeString(alt) + "</code>"
	}
	src := ghRaw + escPath(repo)
	if rest, served := strings.CutPrefix(repo, "site/"); served {
		src = "../" + escPath(rest)
	}
	return fmt.Sprintf("<img src=\"%s%s\"%s loading=\"lazy\">", html.EscapeString(src), html.EscapeString(frag), attrs)
}

// ---------------------------------------------------------------- test counts

// testRunsPath is the one recorded run file every count in the docs comes from.
// To refresh the counts, edit it after a run and execute sitegen.
const testRunsPath = "docs/data/test-runs.json"

// countFiles are the sources that carry count markers.
var countFiles = []string{"README.md", "docs/status.md", "docs/backends.md", "site/index.html"}

type workspaceRun struct {
	Recorded string `json:"recorded"`
	Commit   string `json:"commit"`
	Host     string `json:"host"`
	Command  string `json:"command"`
	Source   string `json:"source"`
	Passed   int    `json:"passed"`
	Failed   int    `json:"failed"`
	Ignored  int    `json:"ignored"`
	Suites   int    `json:"suites"`
}

type crateCount struct {
	Name    string `json:"name"`
	Passed  int    `json:"passed"`
	Ignored int    `json:"ignored"`
	Note    string `json:"note,omitempty"`
}

type perCrateRun struct {
	Recorded string       `json:"recorded"`
	Commit   string       `json:"commit,omitempty"`
	Host     string       `json:"host"`
	Command  string       `json:"command"`
	Source   string       `json:"source"`
	Crates   []crateCount `json:"crates"`
	Go       struct {
		Passed  int    `json:"passed"`
		Failed  int    `json:"failed"`
		Command string `json:"command"`
	} `json:"go"`
}

// testRuns is the schema of docs/data/test-runs.json: the latest full-suite run
// and the latest run that recorded per-crate counts. They may be different runs;
// each carries its own date so the docs can say which numbers are older.
type testRuns struct {
	Workspace workspaceRun `json:"workspace"`
	PerCrate  perCrateRun  `json:"per_crate"`
}

func loadTestRuns(p string) (*testRuns, error) {
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	return parseTestRuns(b, p)
}

var reCommit = regexp.MustCompile(`^[0-9a-f]{7,40}$`)

// requiredRunKeys lists, per object path, the keys that must be present: a
// missing count must not read as zero.
var requiredRunKeys = map[string][]string{
	"":                   {"workspace", "per_crate"},
	"workspace":          {"recorded", "commit", "host", "command", "source", "passed", "failed", "ignored", "suites"},
	"per_crate":          {"recorded", "host", "command", "source", "crates", "go"},
	"per_crate.crates[]": {"name", "passed"},
	"per_crate.go":       {"passed", "failed", "command"},
}

// checkRunShape rejects duplicate keys (encoding/json keeps the last one) and
// absent required keys, by walking the token stream.
func checkRunShape(b []byte) error {
	dec := json.NewDecoder(bytes.NewReader(b))
	return walkRunObject(dec, "")
}

func walkRunValue(dec *json.Decoder, path string) error {
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	switch d := tok.(type) {
	case json.Delim:
		if d == '{' {
			return walkRunMembers(dec, path)
		}
		for dec.More() { // array: elements share the path of their array
			if tok, err := dec.Token(); err != nil {
				return err
			} else if e, ok := tok.(json.Delim); ok && e == '{' {
				if err := walkRunMembers(dec, path+"[]"); err != nil {
					return err
				}
			} else if ok {
				return fmt.Errorf("unexpected nested array at %q", path)
			}
		}
		_, err := dec.Token()
		return err
	}
	return nil
}

func walkRunObject(dec *json.Decoder, path string) error {
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return fmt.Errorf("the run file must be a JSON object")
	}
	return walkRunMembers(dec, path)
}

func walkRunMembers(dec *json.Decoder, path string) error {
	seen := map[string]bool{}
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return err
		}
		key, _ := tok.(string)
		if seen[key] {
			return fmt.Errorf("duplicate key %q in %q", key, path)
		}
		seen[key] = true
		child := key
		if path != "" {
			child = path + "." + key
		}
		if err := walkRunValue(dec, child); err != nil {
			return err
		}
	}
	if _, err := dec.Token(); err != nil {
		return err
	}
	for _, k := range requiredRunKeys[path] {
		if !seen[k] {
			return fmt.Errorf("missing required key %q in %q", k, path)
		}
	}
	return nil
}

func parseTestRuns(b []byte, name string) (*testRuns, error) {
	if err := checkRunShape(b); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var t testRuns
	if err := dec.Decode(&t); err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	if dec.More() {
		return nil, fmt.Errorf("%s: trailing data after the run object", name)
	}
	w := t.Workspace
	if !reCommit.MatchString(w.Commit) || (t.PerCrate.Commit != "" && !reCommit.MatchString(t.PerCrate.Commit)) {
		return nil, fmt.Errorf("%s: commit must be 7 to 40 lowercase hex digits", name)
	}
	if w.Recorded == "" || w.Commit == "" || w.Command == "" || w.Source == "" || w.Host == "" {
		return nil, fmt.Errorf("%s: workspace needs recorded, commit, host, command and source", name)
	}
	if w.Passed <= 0 || w.Failed < 0 || w.Ignored < 0 || w.Suites <= 0 {
		return nil, fmt.Errorf("%s: workspace counts must be positive (failed and ignored may be 0)", name)
	}
	c := t.PerCrate
	if c.Recorded == "" || c.Command == "" || c.Source == "" || c.Host == "" || len(c.Crates) == 0 {
		return nil, fmt.Errorf("%s: per_crate needs recorded, host, command, source and crates", name)
	}
	seen := map[string]bool{}
	for _, cr := range c.Crates {
		if cr.Name == "" || seen[cr.Name] || cr.Passed < 0 || cr.Ignored < 0 {
			return nil, fmt.Errorf("%s: per_crate crate %q is empty, duplicated or negative", name, cr.Name)
		}
		seen[cr.Name] = true
	}
	if c.Go.Passed < 0 || c.Go.Failed < 0 || c.Go.Command == "" {
		return nil, fmt.Errorf("%s: per_crate go needs a command and non-negative counts", name)
	}
	return &t, nil
}

func (c crateCount) text() string {
	s := fmt.Sprintf("%d passed", c.Passed)
	if c.Ignored > 0 {
		s += fmt.Sprintf(", %d ignored", c.Ignored)
	}
	if c.Note != "" {
		s += " " + c.Note
	}
	return s
}

func (t *testRuns) sumPassed() int {
	n := 0
	for _, c := range t.PerCrate.Crates {
		n += c.Passed
	}
	return n
}

// value resolves a count key to its text. Block keys (crates.list) return
// markdown wrapped in blank lines.
func (t *testRuns) value(key string) (string, error) {
	w, c := t.Workspace, t.PerCrate
	switch key {
	case "ws.passed":
		return fmt.Sprint(w.Passed), nil
	case "ws.failed":
		return fmt.Sprint(w.Failed), nil
	case "ws.ignored":
		return fmt.Sprint(w.Ignored), nil
	case "ws.suites":
		return fmt.Sprint(w.Suites), nil
	case "ws.date":
		return w.Recorded, nil
	case "ws.commit":
		return w.Commit, nil
	case "ws.line":
		return fmt.Sprintf("%d passed, %d failed, %d ignored, %d suites", w.Passed, w.Failed, w.Ignored, w.Suites), nil
	case "ws.summary":
		return fmt.Sprintf("%d passed, %d failed, %d ignored, %d suites (`%s` at `%s`, %s, %s)", w.Passed, w.Failed, w.Ignored, w.Suites, w.Command, w.Commit, w.Recorded, w.Host), nil
	case "crates.date":
		return c.Recorded, nil
	case "crates.sum":
		return fmt.Sprint(t.sumPassed()), nil
	case "crates.command":
		return "`" + c.Command + "`", nil
	case "go.passed":
		return fmt.Sprint(c.Go.Passed), nil
	case "go.failed":
		return fmt.Sprint(c.Go.Failed), nil
	case "go.command":
		return "`" + c.Go.Command + "`", nil
	case "crates.list":
		var b strings.Builder
		b.WriteString("\n\n")
		for _, cr := range c.Crates {
			fmt.Fprintf(&b, "- `%s`: %s\n", cr.Name, cr.text())
		}
		b.WriteString("\n")
		return b.String(), nil
	}
	if name, ok := strings.CutPrefix(key, "crate."); ok {
		bare := false
		if n, ok := strings.CutSuffix(name, ".passed"); ok {
			name, bare = n, true
		}
		for _, cr := range c.Crates {
			if cr.Name == name {
				if bare {
					return fmt.Sprint(cr.Passed), nil
				}
				return cr.text(), nil
			}
		}
		return "", fmt.Errorf("count key %q: no crate %q in %s", key, name, testRunsPath)
	}
	return "", fmt.Errorf("unknown count key %q", key)
}

// countRe matches <!--count:KEY-->old text<!--/count-->. The comments are
// invisible in rendered markdown and HTML; the text between them is rewritten.
var countRe = regexp.MustCompile(`(?s)<!--count:([A-Za-z0-9._-]+)-->(.*?)<!--/count-->`)

// countTokenRe finds anything that looks like a count marker, well formed or not,
// so a malformed one is an error rather than an ignored comment.
var countTokenRe = regexp.MustCompile(`(?s)<!--\s*/?\s*count\b.*?-->`)

var (
	countOpenRe  = regexp.MustCompile(`^<!--count:([A-Za-z0-9._-]+)-->$`)
	countCloseTk = "<!--/count-->"
)

// validateCountSpans requires markers to alternate open, close, exactly spelled.
func validateCountSpans(text string) error {
	open := ""
	for _, tok := range countTokenRe.FindAllString(text, -1) {
		switch {
		case countOpenRe.MatchString(tok):
			if open != "" {
				return fmt.Errorf("count span %q is opened inside the unclosed span %q", tok, open)
			}
			open = tok
		case tok == countCloseTk:
			if open == "" {
				return fmt.Errorf("stray %s with no open marker", countCloseTk)
			}
			open = ""
		default:
			return fmt.Errorf("malformed count marker %q: write <!--count:KEY--> ... %s", tok, countCloseTk)
		}
	}
	if open != "" {
		return fmt.Errorf("count span %q is never closed", open)
	}
	return nil
}

// applyCounts rewrites every marked span from the run file (markdown text) and
// returns the new text and how many spans it saw.
func applyCounts(text string, t *testRuns) (string, int, error) {
	return applyCountsWith(text, t, func(v string) string { return v })
}

// applyCountsWith is applyCounts with an escaper for the inserted values, so an
// HTML source never receives a raw value.
func applyCountsWith(text string, t *testRuns, esc func(string) string) (string, int, error) {
	if err := validateCountSpans(text); err != nil {
		return "", 0, err
	}
	n := 0
	var firstErr error
	out := countRe.ReplaceAllStringFunc(text, func(m string) string {
		key := countRe.FindStringSubmatch(m)[1]
		v, err := t.value(key)
		if err != nil {
			if firstErr == nil {
				firstErr = err
			}
			return m
		}
		n++
		return "<!--count:" + key + "-->" + esc(v) + "<!--/count-->"
	})
	return out, n, firstErr
}

// ---------------------------------------------------------------- pages

func catTitle(key string) string {
	for _, c := range categories {
		if c.Key == key {
			return c.Title
		}
	}
	return key
}

const tailwindConfig = `tailwind.config={darkMode:'class',theme:{extend:{colors:{brand:{50:'#fff1f2',100:'#ffe4e6',200:'#fecdd3',300:'#fda4af',400:'#fb7185',500:'#f43f5e',600:'#e11d48',700:'#be123c',800:'#9f1239',900:'#881337',950:'#4c0519',crimson:'#ff2d55',ruby:'#d90429',flame:'#ff4b2b',glow:'#ff3366'}},fontFamily:{sans:['"Plus Jakarta Sans"','system-ui','-apple-system','sans-serif'],mono:['"Fira Code"','ui-monospace','monospace']},boxShadow:{'glow-red':'0 0 35px -5px rgba(255,45,85,0.35)','glow-subtle':'0 0 20px -5px rgba(255,45,85,0.15)'}}}}`

func jsonQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func headHTML(title, desc, slug string) string {
	t := html.EscapeString(title)
	d := html.EscapeString(desc)
	canon := siteURL + "/reference/"
	if slug != "" {
		canon += slug + ".html"
	}
	ld := fmt.Sprintf(`{
    "@context": "https://schema.org",
    "@type": "TechArticle",
    "headline": %s,
    "description": %s,
    "url": %s,
    "image": "%s/assets/ojas-og.png",
    "author": {
      "@type": "Person",
      "name": "Bharath Chandra Vaddaram",
      "url": "https://bharath.vbcr.dev/"
    },
    "publisher": {
      "@type": "Organization",
      "name": "ojas",
      "url": "%s"
    }
  }`, jsonQuote(title+" · ojas reference"), jsonQuote(desc), jsonQuote(canon), siteURL, siteURL)
	ldSafe := strings.ReplaceAll(ld, "</", "<\\/")

	return `<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>` + t + ` · ojas reference</title>
  <meta name="description" content="` + d + `">
  <meta name="keywords" content="ojas, ` + strings.ToLower(t) + `, rust deep learning, machine learning, pytorch alternative, neural network, autograd, tensors, metal, wgpu">
  <meta name="author" content="Bharath Chandra Vaddaram">
  <meta name="robots" content="index, follow">
  <meta name="theme-color" content="#08080a">
  <link rel="canonical" href="` + canon + `">
  <meta property="og:type" content="article">
  <meta property="og:site_name" content="ojas">
  <meta property="og:url" content="` + canon + `">
  <meta property="og:title" content="` + t + ` · ojas reference">
  <meta property="og:description" content="` + d + `">
  <meta property="og:image" content="` + siteURL + `/assets/ojas-og.png">
  <meta property="og:image:width" content="1200">
  <meta property="og:image:height" content="630">
  <meta property="og:image:alt" content="` + t + ` · ojas reference">
  <meta property="og:locale" content="en_US">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="` + t + ` · ojas reference">
  <meta name="twitter:description" content="` + d + `">
  <meta name="twitter:image" content="` + siteURL + `/assets/ojas-og.png">
  <meta name="twitter:image:alt" content="` + t + ` · ojas reference">
  <link rel="icon" href="../assets/favicon-32.png" type="image/png" sizes="32x32">
  <link rel="apple-touch-icon" href="../assets/apple-touch-icon.png" sizes="180x180">
  <link rel="manifest" href="../site.webmanifest">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <script>` + tailwindConfig + `</script>
  <link rel="stylesheet" href="../assets/ojas.css">
  <link rel="stylesheet" href="../assets/reference.css">
  <script type="application/ld+json">
  ` + ldSafe + `
  </script>
</head>
`
}

func topBar() string {
	return `<body class="font-sans antialiased bg-[#08080a] text-slate-100 min-h-screen relative bg-grid">
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="sticky top-0 z-50 glass-nav">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-3">
      <a href="../" class="flex items-center gap-3 shrink-0">
        <img src="../assets/ojas-tile-512.png" alt="ojas" class="w-9 h-9" width="36" height="36">
        <span class="text-xl font-extrabold tracking-wider font-mono text-gradient-red">ojas</span>
      </a>
      <nav class="flex items-center gap-1 text-sm" aria-label="Site">
        <a href="../#labs" class="hidden sm:inline px-2.5 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-white/5 transition">Labs</a>
        <a href="../#benchmarks" class="hidden sm:inline px-2.5 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-white/5 transition">Benchmarks</a>
        <a href="../#guide" class="px-2.5 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-white/5 transition">Guide</a>
        <a href="./" class="px-2.5 py-1.5 rounded-lg text-white bg-white/10 transition" aria-current="page">Reference</a>
        <a href="https://github.com/bharathvbcr/ojas" rel="noopener" class="px-2.5 py-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-white/5 transition">GitHub</a>
      </nav>
    </div>
  </header>
`
}

const footer = `  <footer class="border-t border-white/10 bg-black py-8 mt-16 text-slate-500 text-xs">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
      <span class="font-mono">ojas · dual licensed MIT OR Apache-2.0</span>
      <span>Pages are rendered from the repository's markdown; <a class="underline hover:text-white" href="https://github.com/bharathvbcr/ojas/tree/main/docs" rel="noopener">the markdown is the source</a>.</span>
    </div>
  </footer>
  <script src="../assets/reference.js"></script>
</body>
</html>
`

func sidebar(docs []*doc, cur string) string {
	var b strings.Builder
	b.WriteString("<nav class=\"ref-nav\" aria-label=\"Reference\">\n<a href=\"./\" class=\"ref-nav-home\">All reference pages</a>\n")
	for _, c := range categories {
		var items []*doc
		for _, d := range docs {
			if d.Cat == c.Key {
				items = append(items, d)
			}
		}
		if len(items) == 0 {
			continue
		}
		fmt.Fprintf(&b, "<p class=\"ref-nav-cat\">%s</p>\n<ul>\n", html.EscapeString(c.Title))
		for _, d := range items {
			cls, aria := "", ""
			if d.Slug == cur {
				cls, aria = ` class="current"`, ` aria-current="page"`
			}
			fmt.Fprintf(&b, "<li><a href=\"%s.html\"%s%s>%s</a></li>\n", d.Slug, cls, aria, html.EscapeString(navLabel(d)))
		}
		b.WriteString("</ul>\n")
	}
	b.WriteString("</nav>\n")
	return b.String()
}

func navLabel(d *doc) string {
	if d.Cat == "crates" {
		if d.Slug == "readme" {
			return "Project README"
		}
		return strings.TrimPrefix(d.Slug, "crate-")
	}
	return d.Title
}

func pageHTML(d *doc, docs []*doc, prev, next *doc) string {
	var b strings.Builder
	b.WriteString(headHTML(d.Title, d.Desc, d.Slug))
	b.WriteString(topBar())
	b.WriteString("  <div class=\"max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 grid lg:grid-cols-[16rem_minmax(0,1fr)] xl:grid-cols-[16rem_minmax(0,1fr)_13rem] gap-8\">\n")
	b.WriteString("    <aside class=\"ref-side\"><details class=\"lg:open-always\" open>\n<summary class=\"lg:hidden\">Reference pages</summary>\n")
	b.WriteString(sidebar(docs, d.Slug))
	b.WriteString("</details></aside>\n")
	b.WriteString("    <main id=\"main\" class=\"min-w-0\">\n")
	fmt.Fprintf(&b, "      <p class=\"ref-crumb\"><a href=\"./\">Reference</a> / %s</p>\n", html.EscapeString(catTitle(d.Cat)))
	b.WriteString("      <article class=\"prose-ojas\">\n")
	b.WriteString(d.Body)
	b.WriteString("      </article>\n")
	fmt.Fprintf(&b, "      <p class=\"ref-source\">Rendered from <a href=\"%s%s\" rel=\"noopener\"><code>%s</code></a>. If this page and the repository disagree, the repository wins.</p>\n",
		ghBlob, html.EscapeString(d.Src), html.EscapeString(d.Src))
	b.WriteString("      <nav class=\"ref-pager\" aria-label=\"Previous and next\">\n")
	if prev != nil {
		fmt.Fprintf(&b, "        <a href=\"%s.html\" rel=\"prev\"><span>Previous</span>%s</a>\n", prev.Slug, html.EscapeString(navLabel(prev)))
	} else {
		b.WriteString("        <span></span>\n")
	}
	if next != nil {
		fmt.Fprintf(&b, "        <a href=\"%s.html\" rel=\"next\" class=\"next\"><span>Next</span>%s</a>\n", next.Slug, html.EscapeString(navLabel(next)))
	}
	b.WriteString("      </nav>\n    </main>\n")
	if len(d.Heads) > 1 {
		b.WriteString("    <nav class=\"ref-toc\" aria-label=\"On this page\"><p>On this page</p><ol>\n")
		for _, h := range d.Heads {
			cls := ""
			if h.Level == 3 {
				cls = ` class="sub"`
			}
			fmt.Fprintf(&b, "<li%s><a href=\"#%s\">%s</a></li>\n", cls, h.ID, html.EscapeString(h.Text))
		}
		b.WriteString("</ol></nav>\n")
	}
	b.WriteString("  </div>\n")
	b.WriteString(footer)
	return b.String()
}

type searchEntry struct {
	Slug  string   `json:"slug"`
	Title string   `json:"title"`
	Cat   string   `json:"cat"`
	Desc  string   `json:"desc"`
	Heads []string `json:"heads"`
}

func indexHTML(docs []*doc) string {
	js := bytes.ReplaceAll(searchJSON(docs), []byte("</"), []byte("<\\/"))

	var b strings.Builder
	b.WriteString(headHTML("Reference", "Design contracts, status, benchmark tables, audits and per-crate guides for ojas, rendered from the repository's own markdown.", ""))
	b.WriteString(topBar())
	b.WriteString("  <main id=\"main\" class=\"max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12\">\n")
	b.WriteString("    <p class=\"ref-crumb\">Reference</p>\n    <h1 class=\"text-3xl sm:text-4xl font-extrabold text-white mb-3\">ojas reference</h1>\n")
	fmt.Fprintf(&b, "    <p class=\"text-slate-400 leading-relaxed max-w-3xl mb-6\">%d pages rendered from the repository's own markdown: the contracts every backend follows, what passes today, the raw benchmark tables, the audits, and a guide for each crate. Each page links to its source; where a page and the repository disagree, the repository wins.</p>\n", len(docs))
	b.WriteString("    <label for=\"refFilter\" class=\"sr-only\">Filter reference pages</label>\n    <input id=\"refFilter\" type=\"search\" placeholder=\"Filter by title or section, e.g. budget, Metal, shape\" class=\"w-full mb-2 px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-brand-500/60\" autocomplete=\"off\">\n")
	b.WriteString("    <p id=\"refCount\" class=\"text-xs font-mono text-slate-500 mb-8\" aria-live=\"polite\"></p>\n")
	for _, c := range categories {
		var items []*doc
		for _, d := range docs {
			if d.Cat == c.Key {
				items = append(items, d)
			}
		}
		if len(items) == 0 {
			continue
		}
		fmt.Fprintf(&b, "    <section class=\"ref-cat\" data-cat=\"%s\">\n      <h2 class=\"text-xl font-bold text-white mb-1\">%s</h2>\n      <p class=\"text-sm text-slate-500 mb-4\">%s</p>\n      <ul class=\"grid sm:grid-cols-2 gap-3\">\n",
			c.Key, html.EscapeString(c.Title), html.EscapeString(c.Blurb))
		for _, d := range items {
			fmt.Fprintf(&b, "        <li data-slug=\"%s\"><a href=\"%s.html\" class=\"ref-card\"><span class=\"ref-card-title\">%s</span><span class=\"ref-card-desc\">%s</span><span class=\"ref-card-meta\">%d sections</span></a></li>\n",
				d.Slug, d.Slug, html.EscapeString(navLabel(d)), html.EscapeString(clip(d.Desc, 150)), len(d.Heads))
		}
		b.WriteString("      </ul>\n    </section>\n")
	}
	b.WriteString("    <p id=\"refEmpty\" class=\"hidden text-slate-400\">No page matches that filter.</p>\n")
	b.WriteString("    <script type=\"application/json\" id=\"refIndex\">" + string(js) + "</script>\n")
	b.WriteString("  </main>\n")
	b.WriteString(footer)
	return b.String()
}
