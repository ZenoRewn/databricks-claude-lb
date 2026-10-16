"""Documentation links resolve. Author: Zeno Ren.

CLAUDE.md asks for documentation to be verified by links, structure and
rendering rather than by prose review, but nothing enforced the first of those:
a renamed heading or a moved file broke a link silently, and `docs/` is the
entry point this project points operators at during an incident. One such break
was already present when this test was written — `docs/TROUBLESHOOTING.md`
linked section 16 with a doubled hyphen that no heading produces.

Scope is `git ls-files '*.md'`, so untracked scratch and the private release
tree under `.release-private/` are excluded. The latter is a partial copy of
`docs/` without the `reviews/` subtree, so its links are expected to dangle;
asserting on them would mean asserting on something that is deliberately not
in the repository.

On anchors, honestly: `_slug` approximates GitHub's `github-slugger` (strip
punctuation, keep letters/digits/CJK, spaces to hyphens). It is not that
algorithm and no copy of it is available offline here. The evidence that it is
close enough is that it agrees with every hand-written anchor in the
repository — the one disagreement it reported was a genuine defect. Where a
heading's punctuation makes its slug non-obvious, link an explicit
`<a id="...">` instead of trusting either implementation; section 16 of
TROUBLESHOOTING.md is the worked example.
"""
import re
import subprocess
import unicodedata
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# `](target)` and `](<target>)`; titles are unused in this repository.
LINK = re.compile(r'\]\(\s*<?([^)\s>]+)>?\s*\)')
HEADING = re.compile(r'^#{1,6}\s+(.*)$', re.M)
EXPLICIT_ANCHOR = re.compile(r'<a\s+(?:id|name)="([^"]+)"')
FENCE = re.compile(r'^```.*?^```', re.M | re.S)


def _tracked_markdown():
    out = subprocess.run(['git', 'ls-files', '*.md'], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return sorted(Path(line) for line in out.split('\n') if line)


def _slug(heading: str) -> str:
    """Approximate github-slugger. See the module docstring for the caveat."""
    text = re.sub(r'`', '', heading.strip()).lower()
    kept = [c for c in text
            if c.isalnum() or c in ' -_' or unicodedata.category(c) == 'Lo']
    return ''.join(kept).replace(' ', '-')


def _anchors(text: str) -> set:
    return ({_slug(m.group(1)) for m in HEADING.finditer(text)}
            | set(EXPLICIT_ANCHOR.findall(text)))


class DocumentationLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = _tracked_markdown()
        # Fenced blocks hold sample output and shell transcripts; a bracket pair
        # in there is not a link the reader can follow.
        cls.bodies = {p: FENCE.sub('', (ROOT / p).read_text()) for p in cls.files}
        cls.anchors = {p: _anchors(cls.bodies[p]) for p in cls.files}

    def test_the_scope_is_not_accidentally_empty(self):
        # Without this the suite would pass by checking nothing if git ls-files
        # ever stops matching.
        self.assertGreater(len(self.files), 20)
        self.assertIn(Path('CLAUDE.md'), self.files)
        self.assertIn(Path('docs/CONFIGURATION.md'), self.files)

    def test_the_private_release_tree_is_out_of_scope(self):
        self.assertEqual([p for p in self.files if '.release-private' in p.parts], [])

    def test_every_relative_target_exists(self):
        missing = []
        for path in self.files:
            for match in LINK.finditer(self.bodies[path]):
                target = match.group(1)
                if target.startswith(('http://', 'https://', 'mailto:', '#')):
                    continue
                file_part = target.partition('#')[0]
                if not (ROOT / path).parent.joinpath(file_part).exists():
                    missing.append(f'{path} -> {target}')
        self.assertEqual(missing, [], 'these documentation links point at nothing')

    def test_no_explicit_anchor_is_declared_twice(self):
        """A repeated id is invalid HTML and makes its links land anywhere.

        Caught the hard way: an explicit anchor was added to a section that
        already had one, and every link test still passed — resolution only
        checks that the name exists, not that it is unique. So the duplicate
        would have shipped, and which copy a browser jumps to is undefined.
        """
        duplicated = []
        for path in self.files:
            seen, repeats = set(), set()
            for name in EXPLICIT_ANCHOR.findall(self.bodies[path]):
                (repeats if name in seen else seen).add(name)
            duplicated += [f'{path} -> #{name}' for name in sorted(repeats)]
        self.assertEqual(duplicated, [],
                         'these explicit anchors are declared more than once in one file')

    def test_every_same_file_anchor_resolves(self):
        broken = []
        for path in self.files:
            for match in LINK.finditer(self.bodies[path]):
                target = match.group(1)
                if not target.startswith('#'):
                    continue
                if target[1:] not in self.anchors[path]:
                    broken.append(f'{path} -> {target}')
        self.assertEqual(broken, [], 'no heading or explicit anchor produces these')

    def test_every_cross_file_anchor_resolves(self):
        broken = []
        for path in self.files:
            for match in LINK.finditer(self.bodies[path]):
                target = match.group(1)
                if target.startswith(('http://', 'https://', 'mailto:', '#')):
                    continue
                file_part, _, anchor = target.partition('#')
                if not anchor:
                    continue
                dest = ((ROOT / path).parent / file_part).resolve()
                try:
                    key = dest.relative_to(ROOT)
                except ValueError:
                    continue
                # Only markdown in scope has a parsed anchor set; a link into an
                # untracked or non-markdown file is covered by the test above.
                if key in self.anchors and anchor not in self.anchors[key]:
                    broken.append(f'{path} -> {target}')
        self.assertEqual(broken, [], 'the file exists but has no such anchor')

    def test_the_review_index_reaches_every_receipt_directory(self):
        """A receipt nobody links to is evidence nobody can find.

        Link resolution only catches links that point at nothing; it says
        nothing about a receipt that no link points *at*. Three directories had
        drifted out that way — `2026-10-01-pricing-aks` was reachable only from
        a stale release paragraph in `docs/AKS.md`, and when that paragraph was
        replaced by a pointer to this index (which did not list it) the receipt
        became unreachable. The other two 2026-10-01 directories had never been
        listed at all.

        `docs/reviews/README.md` is what `docs/AKS.md` and `README.md` now name
        as the single place release history is maintained, so its coverage has
        to be enforced rather than remembered.
        """
        index = ROOT / 'docs' / 'reviews' / 'README.md'
        body = index.read_text()
        directories = sorted(p.name for p in index.parent.iterdir()
                             if p.is_dir() and not p.name.startswith('.'))
        self.assertGreater(len(directories), 20, 'receipt tree unexpectedly small')
        unreferenced = [name for name in directories if name not in body]
        self.assertEqual(unreferenced, [],
                         'these receipts exist but the review index links none of them')

    def test_the_api_table_entry_point_still_links_the_documents_it_cites(self):
        # CLAUDE.md is the collaboration entry point; if its own links rot the
        # reader has no route into the rest of the documentation.
        body = self.bodies[Path('CLAUDE.md')]
        cited = {m.group(1).partition('#')[0] for m in LINK.finditer(body)
                 if not m.group(1).startswith(('http', 'mailto:', '#'))}
        self.assertIn('docs/RESILIENCE.md', cited)
        self.assertIn('README.md', cited)
        for target in cited:
            with self.subTest(target=target):
                self.assertTrue((ROOT / target).exists())


class SlugApproximationTests(unittest.TestCase):
    """Pin the slug rules this module relies on, so a change is deliberate."""

    def test_ascii_headings_lowercase_and_hyphenate(self):
        self.assertEqual(_slug('Release Tool'), 'release-tool')

    def test_punctuation_is_dropped_not_turned_into_hyphens(self):
        # This is the rule the broken section-16 link got wrong: it wrote `--`
        # where the em dash had been, but punctuation leaves nothing behind.
        self.assertEqual(_slug('16. 「状态失效」—— 实测不成立'),
                         '16-状态失效-实测不成立')

    def test_full_width_punctuation_is_dropped_too(self):
        self.assertEqual(_slug('策略降级：配置是基线，存储是 overlay'),
                         '策略降级配置是基线存储是-overlay')

    def test_inline_code_markers_do_not_reach_the_slug(self):
        # A doubled hyphen *can* be right: here one comes from `claude-` and one
        # from the space left after `*` was dropped. So `--` in an anchor is not
        # by itself the defect — the defect is assuming punctuation becomes one.
        self.assertEqual(_slug('`claude-*` 走 GHCP'), 'claude--走-ghcp')


if __name__ == '__main__':
    unittest.main()
