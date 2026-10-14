package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Run from the repository root:
//
//	go test scripts/sitegen/main.go scripts/sitegen/main_test.go

const fixtureRuns = `{
  "workspace": {"recorded": "2026-10-09", "commit": "abc1234", "host": "macOS", "command": "cargo test --workspace", "source": "fixture", "passed": 100, "failed": 1, "ignored": 4, "suites": 9},
  "per_crate": {
    "recorded": "2026-10-05", "host": "macOS", "command": "cargo test -p <crate>", "source": "fixture",
    "crates": [
      {"name": "ojas-a", "passed": 7, "ignored": 2},
      {"name": "ojas-b", "passed": 5, "note": "(feature off)"}
    ],
    "go": {"passed": 3, "failed": 0, "command": "go test ./..."}
  }
}`

func mustRuns(t *testing.T) *testRuns {
	t.Helper()
	r, err := parseTestRuns([]byte(fixtureRuns), "fixture")
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func newRenderer(src string, tracked ...string) *renderer {
	ts := trackedSet{}
	for _, f := range tracked {
		ts[f] = false
		for d := filepath.ToSlash(filepath.Dir(f)); d != "."; d = filepath.ToSlash(filepath.Dir(d)) {
			ts[d] = true
		}
	}
	return &renderer{tracked: ts, doc: &doc{Slug: "x", Src: src}, bySrc: map[string]*doc{}, ids: map[string]int{}}
}

func TestImageRendering(t *testing.T) {
	tracked := []string{"docs/brand/logo.png", "docs/x.md", "site/assets/ünï.svg"}
	cases := []struct{ name, in, want string }{
		{"site image is served by the site", "![chart](../site/assets/ünï.svg)",
			`<img src="../assets/ünï.svg" alt="chart" loading="lazy">`},
		{"title", `![chart](../site/assets/ünï.svg "Chart title")`,
			`<img src="../assets/ünï.svg" alt="chart" title="Chart title" loading="lazy">`},
		{"non-site tracked file loads from GitHub raw", "![logo](brand/logo.png)",
			`<img src="` + ghRaw + `docs/brand/logo.png" alt="logo" loading="lazy">`},
		{"external", "![x](https://example.com/a.png)", `<img src="https://example.com/a.png" alt="x" loading="lazy">`},
		{"alt is escaped", `![a <b> & "q"](../site/assets/ünï.svg)`,
			`<img src="../assets/ünï.svg" alt="a &lt;b&gt; &amp; &#34;q&#34;" loading="lazy">`},
		{"image inside a link keeps both", "[![logo](brand/logo.png)](x.md)",
			`<a href="` + ghBlob + `docs/x.md" rel="noopener"><img src="` + ghRaw + `docs/brand/logo.png" alt="logo" loading="lazy"></a>`},
		{"no literal bang", "see ![chart](../site/assets/ünï.svg) here",
			`see <img src="../assets/ünï.svg" alt="chart" loading="lazy"> here`},
		{"code span is untouched", "`![a](b.svg)`", "<code>![a](b.svg)</code>"},
	}
	for _, c := range cases {
		r := newRenderer("docs/page.md", tracked...)
		if got := r.inline(c.in); got != c.want {
			t.Errorf("%s:\n got  %s\n want %s", c.name, got, c.want)
		}
		if r.unresolved != 0 {
			t.Errorf("%s: unresolved = %d", c.name, r.unresolved)
		}
	}
}

func TestImageProblemsAreCounted(t *testing.T) {
	for _, in := range []string{
		"![x](missing.png)",
		"![x](../site/assets)", // a directory
		"![x](file:///Users/bharath/a.png)",
		"![x](/Users/bharath/a.png)",
		`![x](..\site\a.png)`,
	} {
		r := newRenderer("docs/page.md", "site/assets/a.png")
		r.inline(in)
		if r.unresolved != 1 {
			t.Errorf("%s: unresolved = %d, want 1", in, r.unresolved)
		}
	}
}

func TestLinksThatOnlyWorkOnOneMachineAreRejected(t *testing.T) {
	for _, target := range []string{
		"file:///Users/bharath/Code/research/ojas/README.md",
		"FILE:///Users/bharath/Code/research/ojas/README.md",
		"file://localhost/etc/passwd",
		"/Users/bharath/Code/research/ojas/README.md",
		"/etc/hosts",
		`docs\architecture.md`,
		`C:\Users\x\a.md`,
	} {
		r := newRenderer("docs/page.md", "README.md", "docs/architecture.md")
		got := r.inline("[t](" + target + ")")
		if r.unresolved != 1 {
			t.Errorf("%s: unresolved = %d, want 1", target, r.unresolved)
		}
		if strings.Contains(got, "<a ") {
			t.Errorf("%s: rendered a link: %s", target, got)
		}
	}
}

func TestRelativeLinksWithAnchorsAndQueries(t *testing.T) {
	r := newRenderer("docs/page.md", "README.md", "docs/page.md", "docs/other.md", "ojas-core/src/lib.rs")
	r.bySrc["docs/other.md"] = &doc{Slug: "other", Src: "docs/other.md"}
	cases := map[string]string{
		"[a](other.md#sec)":                    `<a href="other.html#sec">a</a>`,
		"[a](../README.md#top)":                `<a href="` + ghBlob + `README.md#top" rel="noopener">a</a>`,
		"[a](../ojas-core/src/lib.rs?plain=1)": `<a href="` + ghBlob + `ojas-core/src/lib.rs?plain=1" rel="noopener">a</a>`,
		"[a](#local)":                          `<a href="#local">a</a>`,
		"[a](https://x.dev/p?q=1#f)":           `<a href="https://x.dev/p?q=1#f" rel="noopener">a</a>`,
		"[a](../ojas-core)":                    `<a href="` + ghTree + `ojas-core" rel="noopener">a</a>`,
	}
	for in, want := range cases {
		if got := r.inline(in); got != want {
			t.Errorf("%s:\n got  %s\n want %s", in, got, want)
		}
	}
	if r.unresolved != 0 {
		t.Errorf("unresolved = %d", r.unresolved)
	}
}

func TestFencedLinksAreNotRendered(t *testing.T) {
	r := newRenderer("docs/page.md")
	out := r.blocks(splitLines("```\n[a](file:///Users/x/y.md) ![i](z.png)\n```"))
	if r.unresolved != 0 {
		t.Fatalf("a link inside a code fence was resolved: unresolved = %d", r.unresolved)
	}
	if !strings.Contains(out, "[a](file:///Users/x/y.md) ![i](z.png)") {
		t.Fatalf("fence content changed: %s", out)
	}
}

func TestCountCommentsAreInvisibleInRenderedMarkdown(t *testing.T) {
	r := newRenderer("README.md")
	out := r.blocks(splitLines("<!--count:ws.passed-->100<!--/count--> tests\n\n<!--count:crates.list-->\n\n- `a`: 1 passed\n\n<!--/count-->\n\ndone"))
	if strings.Contains(out, "&lt;!--") || strings.Contains(out, "<!--") {
		t.Fatalf("a count marker leaked into the page: %s", out)
	}
	if !strings.Contains(out, "<p>100 tests</p>") || !strings.Contains(out, "<li><code>a</code>: 1 passed") || !strings.Contains(out, "<p>done</p>") {
		t.Fatalf("unexpected output: %s", out)
	}
}

func TestWorkspaceMembers(t *testing.T) {
	got, err := workspaceMembers(`[workspace]
resolver = "2"
members = [
    "ojas-core",
    "ojas-cpu", # trailing comment with "quotes"
    # "ojas-removed",
    "ojas-model",
]
exclude = ["bench"]
`)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got, ",") != "ojas-core,ojas-cpu,ojas-model" {
		t.Fatalf("members = %v", got)
	}
	if _, err := workspaceMembers("[package]\nname = \"x\"\n"); err == nil {
		t.Fatal("a manifest without members must fail, not yield no crates")
	}
}

// scratchRepo writes files into a fresh git repository and makes it the working
// directory, the way sitegen runs from the repository root.
func scratchRepo(t *testing.T, files map[string]string) {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		p := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, args := range [][]string{{"init", "-q"}, {"add", "-A"}} {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	t.Chdir(dir)
}

func baseFiles() map[string]string {
	return map[string]string{
		"Cargo.toml":               "[workspace]\nmembers = [\"ojas-a\"]\n",
		"docs/data/test-runs.json": fixtureRuns,
		"README.md":                "# Project\n\nTests: <!--count:ws.passed-->0<!--/count-->.\n",
		"docs/status.md":           "# Status\n\n<!--count:ws.line-->0<!--/count-->\n",
		"docs/backends.md":         "# Backends\n\n| a | <!--count:crate.ojas-a-->0<!--/count--> |\n| - | - |\n",
		"site/index.html":          "<p><!--count:ws.passed-->0<!--/count--> and <!--count:go.passed-->0<!--/count--></p>\n",
		"ojas-a/README.md":         "# ojas-a\n\nA workspace crate with a long enough description to become the page summary.\n",
		"bench/README.md":          "# bench\n\nNot a workspace member, so it gets no crate page of its own here.\n",
		"go/README.md":             "# go\n\nNot a workspace member either, so no crate page is generated for it.\n",
		"gemma-metal/README.md":    "# gemma-metal\n\nAlso not a workspace member, so it is reached by a GitHub link only.\n",
		"site/assets/plots/p.svg":  "<svg/>",
	}
}

func TestOnlyWorkspaceMembersGetCratePages(t *testing.T) {
	scratchRepo(t, baseFiles())
	tracked, err := loadTracked()
	if err != nil {
		t.Fatal(err)
	}
	members, err := workspaceMembersFile("Cargo.toml")
	if err != nil {
		t.Fatal(err)
	}
	docs, err := collect(tracked, members)
	if err != nil {
		t.Fatal(err)
	}
	var slugs []string
	for _, d := range docs {
		slugs = append(slugs, d.Slug)
	}
	got := strings.Join(slugs, ",")
	for _, bad := range []string{"crate-bench", "crate-go", "crate-gemma-metal"} {
		if strings.Contains(got, bad) {
			t.Errorf("%s must not be a crate page: %s", bad, got)
		}
	}
	for _, want := range []string{"crate-ojas-a", "readme", "status", "backends"} {
		if !strings.Contains(got, want) {
			t.Errorf("missing page %s: %s", want, got)
		}
	}
}

func TestRunWritesThenChecks(t *testing.T) {
	scratchRepo(t, baseFiles())
	if err := run(true); err == nil {
		t.Fatal("-check passed on a tree whose generated files are missing")
	}
	if err := run(false); err != nil {
		t.Fatal(err)
	}
	if err := run(true); err != nil {
		t.Fatalf("-check failed right after generating: %v", err)
	}
	for _, p := range []string{"docs/reference", "docs/index.html", "docs/assets"} {
		if _, err := os.Stat(p); err == nil {
			t.Errorf("%s was created: docs/ must not carry a mirror of site/", p)
		}
	}
	b, err := os.ReadFile("site/index.html")
	if err != nil {
		t.Fatal(err)
	}
	if want := "<p><!--count:ws.passed-->100<!--/count--> and <!--count:go.passed-->3<!--/count--></p>\n"; string(b) != want {
		t.Fatalf("site/index.html = %q, want %q", b, want)
	}
}

func TestCheckValidatesSiteIndexNumbers(t *testing.T) {
	scratchRepo(t, baseFiles())
	if err := run(false); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile("site/index.html")
	if err := os.WriteFile("site/index.html", []byte(strings.Replace(string(b), ">100<", ">1510<", 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	err := run(true)
	if err == nil || !strings.Contains(err.Error(), "site/index.html") {
		t.Fatalf("a hand-edited count in site/index.html must fail -check, got %v", err)
	}
	if err := run(false); err != nil {
		t.Fatal(err)
	}
	if err := run(true); err != nil {
		t.Fatalf("regenerating did not repair site/index.html: %v", err)
	}
}

func TestCheckFailsOnFileLinksAndAbsolutePaths(t *testing.T) {
	for _, link := range []string{
		"file:///Users/bharath/Code/research/ojas/ojas-a",
		"/Users/bharath/Code/research/ojas/ojas-a",
	} {
		files := baseFiles()
		files["README.md"] += "\nSee [the crate](" + link + ").\n"
		scratchRepo(t, files)
		for _, check := range []bool{true, false} {
			err := run(check)
			if err == nil || !strings.Contains(err.Error(), "point at nothing") {
				t.Errorf("%s (check=%v): got %v, want a failure", link, check, err)
			}
		}
	}
}

func TestMarkerWithoutCountsIsAnError(t *testing.T) {
	files := baseFiles()
	files["site/index.html"] = "<p>1510 tests, typed by hand</p>\n"
	scratchRepo(t, files)
	err := run(false)
	if err == nil || !strings.Contains(err.Error(), "no count markers") {
		t.Fatalf("got %v", err)
	}
}

const countsInput = "a <!--count:ws.passed-->X<!--/count--> b <!--count:ws.line-->Y<!--/count-->\n" +
	"| <!--count:crate.ojas-a-->?<!--/count--> | <!--count:crate.ojas-b-->?<!--/count--> | <!--count:crate.ojas-a.passed-->?<!--/count-->\n" +
	"<!--count:crates.list-->\nstale\n<!--/count-->\n" +
	"<!--count:crates.sum-->0<!--/count--> <!--count:go.passed-->0<!--/count--> <!--count:ws.date-->d<!--/count--> <!--count:crates.date-->d<!--/count-->\n"

const countsGolden = "a <!--count:ws.passed-->100<!--/count--> b <!--count:ws.line-->100 passed, 1 failed, 4 ignored, 9 suites<!--/count-->\n" +
	"| <!--count:crate.ojas-a-->7 passed, 2 ignored<!--/count--> | <!--count:crate.ojas-b-->5 passed (feature off)<!--/count--> | <!--count:crate.ojas-a.passed-->7<!--/count-->\n" +
	"<!--count:crates.list-->\n\n- `ojas-a`: 7 passed, 2 ignored\n- `ojas-b`: 5 passed (feature off)\n\n<!--/count-->\n" +
	"<!--count:crates.sum-->12<!--/count--> <!--count:go.passed-->3<!--/count--> <!--count:ws.date-->2026-10-09<!--/count--> <!--count:crates.date-->2026-10-05<!--/count-->\n"

func TestCountBlockGenerationGolden(t *testing.T) {
	got, n, err := applyCounts(countsInput, mustRuns(t))
	if err != nil {
		t.Fatal(err)
	}
	if n != 10 {
		t.Errorf("spans = %d, want 10", n)
	}
	if got != countsGolden {
		t.Errorf("got:\n%s\nwant:\n%s", got, countsGolden)
	}
	again, _, err := applyCounts(got, mustRuns(t))
	if err != nil || again != got {
		t.Errorf("generation is not idempotent: %v", err)
	}
}

func TestCountKeysFailLoudly(t *testing.T) {
	for _, in := range []string{
		"<!--count:ws.nope-->1<!--/count-->",
		"<!--count:crate.ojas-missing-->1<!--/count-->",
		"<!--count:crate.ojas-missing.passed-->1<!--/count-->",
	} {
		if _, _, err := applyCounts(in, mustRuns(t)); err == nil {
			t.Errorf("%s: no error", in)
		}
	}
}

func TestParseTestRunsRejectsBadFiles(t *testing.T) {
	good := fixtureRuns
	bad := map[string]string{
		"unknown field":   strings.Replace(good, `"suites": 9`, `"suites": 9, "extra": 1`, 1),
		"zero passed":     strings.Replace(good, `"passed": 100`, `"passed": 0`, 1),
		"missing commit":  strings.Replace(good, `"commit": "abc1234", `, ``, 1),
		"duplicate crate": strings.Replace(good, `"ojas-b"`, `"ojas-a"`, 1),
		"negative":        strings.Replace(good, `"ignored": 4`, `"ignored": -4`, 1),
		"trailing data":   good + `{}`,
		"not json":        `passed: 100`,
	}
	for name, body := range bad {
		if _, err := parseTestRuns([]byte(body), name); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestRecordedRunFileIsValid(t *testing.T) {
	t.Chdir("../..")
	r, err := loadTestRuns(testRunsPath)
	if err != nil {
		t.Fatal(err)
	}
	if r.sumPassed() <= 0 {
		t.Fatal("recorded run file has no per-crate counts")
	}
	for _, p := range countFiles {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if _, n, err := applyCounts(string(raw), r); err != nil || n == 0 {
			t.Errorf("%s: spans=%d err=%v", p, n, err)
		}
	}
}

func TestLinkToUntrackedFileIsRejected(t *testing.T) {
	// A file ignored by git exists on a developer's disk and not in CI, so the
	// link must fail here too.
	r := newRenderer("docs/page.md", "docs/page.md")
	got := r.inline("[a](../target-lane/notes.md)")
	if r.unresolved != 1 || strings.Contains(got, "<a ") {
		t.Fatalf("unresolved = %d, rendered %s", r.unresolved, got)
	}
}
