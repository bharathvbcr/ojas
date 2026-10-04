// sitegen renders the repository's markdown docs and crate READMEs into the
// static reference library under site/reference/.
//
// The markdown files are the source of truth. Run it from the repository root:
//
//	go run scripts/sitegen/main.go          # write site/reference/
//	go run scripts/sitegen/main.go -check   # exit 1 if site/reference/ is stale
//
// It uses only the standard library.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"html"
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
	absRoot = "/Users/bharath/Code/research/ojas/"
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
	{"crates", "Crates", "One README per workspace crate, plus the Go package and the bench harness."},
}

// docCategory maps docs/<slug>.md to a category. A doc missing from the map is
// an error, so a new file cannot be published uncategorised by accident.
var docCategory = map[string]string{
	"architecture": "design", "framework-design": "design", "shape-contract": "design",
	"dtype-policy": "design", "op-coverage": "design", "backends": "design",
	"checkpoint-v1": "design", "adaptive-resources": "design",
	"status": "status", "baseline": "status", "typed-storage-plan": "status",
	"pytorch-parity-plan": "status", "cuda-backend-scoping": "status", "metal-deferred-faults": "status",
	"bench-cpu-vs-torch": "bench", "bench-gpu-vs-torch": "bench",
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
	check := flag.Bool("check", false, "verify site/reference is up to date instead of writing it")
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
	docs, err := collect(tracked)
	if err != nil {
		return err
	}
	bySrc := map[string]*doc{}
	for _, d := range docs {
		bySrc[d.Src] = d
	}
	unresolved := 0
	for _, d := range docs {
		src, err := os.ReadFile(d.Src)
		if err != nil {
			return err
		}
		r := &renderer{tracked: tracked, doc: d, bySrc: bySrc, ids: map[string]int{}}
		d.Body = r.blocks(splitLines(string(src)))
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
	files := map[string][]byte{}
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
	// docs/ is a full local-preview mirror of site/.
	mirrored := map[string][]byte{}
	for name, b := range files {
		mirrored["docs/"+strings.TrimPrefix(name, "site/")] = b
	}
	for name, b := range mirrored {
		files[name] = b
	}
	for _, f := range staticMirror {
		b, err := os.ReadFile("site/" + f)
		if err != nil {
			return err
		}
		files["docs/"+f] = b
	}

	if check {
		return verify(files)
	}
	for name, b := range files {
		if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(name, b, 0o644); err != nil {
			return err
		}
	}
	for _, f := range orphans(files) {
		if err := os.Remove(f); err != nil {
			return err
		}
	}
	fmt.Printf("sitegen: wrote %d files (%d reference pages in site/reference and docs/reference)\n", len(files), len(docs)+1)
	return nil
}

// staticMirror lists hand-written site files copied unchanged into docs/.
var staticMirror = []string{
	"index.html", "robots.txt", "site.webmanifest",
	"assets/ojas.js", "assets/ojas.css", "assets/reference.css", "assets/reference.js",
}

// orphans lists generated pages on disk that no longer have a source.
func orphans(files map[string][]byte) []string {
	var out []string
	for _, dir := range []string{"site/reference", "docs/reference"} {
		existing, _ := filepath.Glob(filepath.Join(dir, "*.html"))
		for _, f := range existing {
			if _, ok := files[filepath.ToSlash(f)]; !ok {
				out = append(out, f)
			}
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
		return fmt.Errorf("generated site files are stale (%d): %s\nrun: go run scripts/sitegen/main.go", len(bad), strings.Join(bad, ", "))
	}
	fmt.Printf("sitegen: site/ and docs/ are up to date (%d files)\n", len(files))
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
// page that a clean checkout (CI) would not have.
func collect(tracked trackedSet) ([]*doc, error) {
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
	crates, _ := filepath.Glob("*/README.md")
	for _, p := range crates {
		dir := filepath.Dir(p)
		if _, ok := tracked[filepath.ToSlash(p)]; !ok || dir == "docs" || dir == "site" {
			continue
		}
		out = append(out, &doc{Slug: "crate-" + dir, Src: filepath.ToSlash(p), Cat: "crates"})
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
	out, err := exec.Command("git", "ls-files").Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-files (needed to validate links against a clean checkout): %w", err)
	}
	t := trackedSet{}
	for _, f := range strings.Split(strings.TrimSpace(string(out)), "\n") {
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
	reHTMLBlock = regexp.MustCompile(`^</?(p|div|img|details|summary|br|table|center)[\s>/]`)
	reHeading   = regexp.MustCompile(`^(#{1,6})\s+(.*?)\s*#*\s*$`)
	reFence     = regexp.MustCompile("^(\\s*)(```+|~~~+)\\s*([A-Za-z0-9_+-]*)")
	reUL        = regexp.MustCompile(`^(\s*)([-*+])\s+(.*)$`)
	reOL        = regexp.MustCompile(`^(\s*)(\d+)[.)]\s+(.*)$`)
	reHR        = regexp.MustCompile(`^\s{0,3}(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$`)
	reTableSp   = regexp.MustCompile(`^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$`)
	reAlert     = regexp.MustCompile(`^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$`)
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
	reCode   = regexp.MustCompile("`+([^`]+?)`+")
	reLink   = regexp.MustCompile(`\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)`)
	reBold   = regexp.MustCompile(`\*\*(.+?)\*\*|__(.+?)__`)
	reItalic = regexp.MustCompile(`(^|[^\w*])\*([^*\s][^*]*?)\*([^\w*]|$)`)
	reAuto   = regexp.MustCompile(`(^|[\s(])(https?://[^\s<)]+[^\s<).,;:])`)
)

func (r *renderer) inline(s string) string {
	var codes []string
	s = reCode.ReplaceAllStringFunc(s, func(m string) string {
		inner := reCode.FindStringSubmatch(m)[1]
		codes = append(codes, "<code>"+html.EscapeString(strings.TrimSpace(inner))+"</code>")
		return fmt.Sprintf("\x00%d\x00", len(codes)-1)
	})
	var links []string
	s = reLink.ReplaceAllStringFunc(s, func(m string) string {
		sm := reLink.FindStringSubmatch(m)
		links = append(links, r.link(sm[1], sm[2]))
		return fmt.Sprintf("\x01%d\x01", len(links)-1)
	})
	s = html.EscapeString(s)
	s = reAuto.ReplaceAllString(s, `$1<a href="$2" rel="noopener">$2</a>`)
	s = reBold.ReplaceAllString(s, "<strong>$1$2</strong>")
	s = reItalic.ReplaceAllString(s, "$1<em>$2</em>$3")
	for i, l := range links {
		s = strings.Replace(s, fmt.Sprintf("\x01%d\x01", i), l, 1)
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
	if k := strings.Index(target, "#"); k >= 0 {
		target, frag = target[:k], target[k:]
	}
	var repo string
	if strings.HasPrefix(target, "file://") {
		t := strings.TrimPrefix(target, "file://")
		if !strings.HasPrefix(t, absRoot) {
			r.unresolved++
			return "<code>" + html.EscapeString(text) + "</code>"
		}
		repo = strings.TrimPrefix(t, absRoot)
	} else {
		repo = path.Clean(path.Join(path.Dir(r.doc.Src), target))
	}
	repo = strings.TrimSuffix(repo, "/")
	if d, ok := r.bySrc[repo]; ok {
		return fmt.Sprintf("<a href=\"%s.html%s\">%s</a>", d.Slug, html.EscapeString(frag), label)
	}
	if repo == "site/index.html" || repo == "docs/index.html" {
		return fmt.Sprintf("<a href=\"../\">%s</a>", label)
	}
	if repo == ".." || strings.HasPrefix(repo, "../") {
		// A sibling checkout (tessl, gusset): not part of this repository, so no page to link to.
		return "<code>" + html.EscapeString(text) + "</code>"
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
		return fmt.Sprintf("<a href=\"%s%s\" rel=\"noopener\">%s</a>", ghTree, html.EscapeString(repo), label)
	}
	return fmt.Sprintf("<a href=\"%s%s%s\" rel=\"noopener\">%s</a>", ghBlob, html.EscapeString(repo), html.EscapeString(frag), label)
}

func (r *renderer) inlineLabel(text string) string { return r.inline(text) }

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

func headHTML(title, desc, slug string) string {
	t := html.EscapeString(title)
	d := html.EscapeString(desc)
	canon := siteURL + "/reference/"
	if slug != "" {
		canon += slug + ".html"
	}
	return `<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>` + t + ` · ojas reference</title>
  <meta name="description" content="` + d + `">
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
