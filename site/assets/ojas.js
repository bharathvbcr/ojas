function $(id) { return document.getElementById(id); }
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
    function wireSegmented(group, onPick) {
      group.querySelectorAll('.seg-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
          group.querySelectorAll('.seg-btn').forEach((b) => b.setAttribute('aria-pressed', b === btn ? 'true' : 'false'));
          onPick(btn.dataset.v ?? btn.dataset.run);
        });
      });
    }

    // Tabs with roving tabindex and arrow keys.
    function wireTabs(list, onSelect) {
      const tabs = Array.from(list.querySelectorAll('[role="tab"]'));
      const select = (tab, focus) => {
        tabs.forEach((t) => {
          const on = t === tab;
          t.setAttribute('aria-selected', on ? 'true' : 'false');
          t.tabIndex = on ? 0 : -1;
        });
        if (focus) tab.focus();
        onSelect(tab);
      };
      tabs.forEach((tab, i) => {
        tab.addEventListener('click', () => select(tab, false));
        tab.addEventListener('keydown', (e) => {
          let next = null;
          if (e.key === 'ArrowRight') next = tabs[(i + 1) % tabs.length];
          if (e.key === 'ArrowLeft') next = tabs[(i - 1 + tabs.length) % tabs.length];
          if (e.key === 'Home') next = tabs[0];
          if (e.key === 'End') next = tabs[tabs.length - 1];
          if (next) { e.preventDefault(); select(next, true); }
        });
      });
      return (tab) => select(tab, false);
    }

function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text).catch(() => legacyCopy(text));
  }
  return legacyCopy(text);
}
function legacyCopy(text) {
  return new Promise((resolve, reject) => {
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.left = '-9999px';
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (err) { ok = false; }
    area.remove();
    if (ok) resolve();
    else reject(new Error('copy failed'));
  });
}
function copyCommand(cmd) {
  copyText(cmd)
    .then(() => showToast('Copied: ' + cmd))
    .catch(() => showToast('Copy failed. Select the command and copy it manually.'));
}
function showToast(msg) {
  const toast = $('toast');
  $('toastMsg').textContent = msg;
  toast.classList.remove('translate-y-20', 'opacity-0', 'pointer-events-none');
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => {
    toast.classList.add('translate-y-20', 'opacity-0', 'pointer-events-none');
  }, 3000);
}
    // --- 5. Focus trap, search, menu ---
    const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled])';
    let trapRoots = null;
    let focusReturn = null;
    function visibleFocusable(root) {
      const found = [];
      if (root.matches && root.matches(FOCUSABLE)) found.push(root);
      root.querySelectorAll(FOCUSABLE).forEach((el) => found.push(el));
      return found.filter((el) => el.getClientRects().length > 0);
    }
    function onTrapKey(e) {
      if (!trapRoots || e.key !== 'Tab') return;
      const list = trapRoots.flatMap((root) => visibleFocusable(root));
      if (!list.length) return;
      const first = list[0];
      const last = list[list.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && (active === first || !list.includes(active))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    }
    function beginTrap(roots, focusEl) {
      if (!focusReturn || focusReturn === document.body) focusReturn = document.activeElement;
      trapRoots = roots;
      document.removeEventListener('keydown', onTrapKey);
      document.addEventListener('keydown', onTrapKey);
      const list = roots.flatMap((root) => visibleFocusable(root));
      const target = focusEl || list[0];
      if (target) target.focus();
    }
    function endTrap(restore) {
      document.removeEventListener('keydown', onTrapKey);
      trapRoots = null;
      if (!restore) return;
      const back = focusReturn;
      focusReturn = null;
      if (back && typeof back.focus === 'function') back.focus();
    }

    const cmdModal = $('cmdModal');
    const cmdInput = $('cmdInput');
    const cmdResults = $('cmdResults');
    let activeCmdIndex = -1;
    const searchCatalog = [
      { title: 'Bit-identical reduction lab', category: 'Home', href: 'index.html#determinism', tab: null, tags: 'determinism bit-identical ieee-754 reproducibility threads exact numerics float sum home' },
      { title: 'Memory budget try_reserve lab', category: 'Home', href: 'index.html?tab=budget#labs', tab: 'tab-budget', tags: 'memory budget capacity refusal oom try_reserve capacityexceeded child kv cache' },
      { title: 'No silent CPU fallback lab', category: 'Home', href: 'index.html?tab=device#labs', tab: 'tab-device', tags: 'device metal wgpu cuda fallback loadon nodevice notcompiled' },
      { title: 'Go panic firewall lab', category: 'Home', href: 'index.html?tab=panic#labs', tab: 'tab-panic', tags: 'panic poison gusset errpanic errpoisoned close go cgo firewall' },
      { title: 'Checked shapes and zero-copy views lab', category: 'Home', href: 'index.html?tab=checked#labs', tab: 'tab-checked', tags: 'overflow checked_mul shape narrow view byte_offset zero-copy window' },
      { title: 'Why ojas, and why I built it', category: 'Why', href: 'why.html', tab: null, tags: 'why motivation author audit tessl metal-native gemma-metal binn lappi gusset mps' },
      { title: 'Questions about the guarantees', category: 'Why', href: 'why.html#faq', tab: null, tags: 'faq determinism budget gusset metal crates.io overflow' },
      { title: 'ojas vs PyTorch and TensorFlow', category: 'Benchmarks', href: 'benchmarks.html#compare', tab: null, tags: 'pytorch tensorflow determinism fallback budget gusset overflow compare' },
      { title: 'CPU vs PyTorch step timings', category: 'Benchmarks', href: 'benchmarks.html#benchmarks', tab: null, tags: 'benchmarks torch wall time apple m5 22.5us 0.455ms breakdown' },
      { title: "What's next", category: 'Roadmap', href: 'roadmap.html', tab: null, tags: 'roadmap next phase framework weights metal attention bf16 cuda what is next' },
      { title: 'Guide overview', category: 'Guide', href: 'guide/index.html', tab: null, tags: 'guide docs developer documentation hierarchy' },
      { title: 'Quickstart: build, test, code samples', category: 'Guide', href: 'guide/quickstart.html', tab: null, tags: 'install quickstart cargo test go test gusset rust tape example loadon step capi' },
      { title: 'Core concepts: tensors, budgets, numerics, errors', category: 'Guide', href: 'guide/concepts.html', tab: null, tags: 'tensor view byte_offset budget child numerics exact fast backend trait ojaserror errors placement unsupported' },
      { title: 'Feature reference', category: 'Guide', href: 'guide/features.html', tab: null, tags: 'features dtype bf16 autograd gradcheck backends metal wgpu adamw muon sampling top-k top-p kv cache safetensors checkpoint bpe' },
      { title: 'Go API reference', category: 'Guide', href: 'guide/go.html', tab: null, tags: 'go api setmodelroot load loadon step generate generategreedy free close errcapacity errnonfinite errdevicelost' },
      { title: 'Architecture and workspace crates', category: 'Guide', href: 'guide/architecture.html', tab: null, tags: 'architecture layers crates ojas-core ojas-cpu ojas-simd ojas-metal ojas-wgpu ojas-autograd ojas-io cuda hip' },
      { title: 'Status and limits', category: 'Guide', href: 'guide/status.html', tab: null, tags: 'status limits nanolab gpt bf16 cuda today' },
      { title: 'Contributing and frozen invariants', category: 'Guide', href: 'guide/contributing.html', tab: null, tags: 'contributing invariants ci_local clippy fmt tests' },
    ];
    function samePath(a, b) {
      const norm = (p) => (p.endsWith('/') ? p + 'index.html' : p);
      return norm(a) === norm(b);
    }
    function goToResult(item) {
      closeCmdModal(false);
      const root = document.body.getAttribute('data-root') || '';
      const url = new URL(root + item.href, location.href);
      if (!samePath(url.pathname, location.pathname)) {
        location.href = url.pathname + url.search + url.hash;
        return;
      }
      if (item.tab) {
        const tab = document.getElementById(item.tab);
        if (tab) tab.click();
      }
      const id = decodeURIComponent(url.hash.replace(/^#/, ''));
      const el = id ? document.getElementById(id) : null;
      if (el) el.scrollIntoView();
      history.replaceState(null, '', url.pathname + url.search + url.hash);
    }
    function filterCmdResults(query) {
      activeCmdIndex = -1;
      const q = (query || '').trim().toLowerCase();
      const filtered = q === '' ? searchCatalog : searchCatalog.filter((item) =>
        item.title.toLowerCase().includes(q) || item.category.toLowerCase().includes(q) || item.tags.includes(q));
      if (!filtered.length) {
        cmdResults.innerHTML = `<div class="p-6 text-center text-slate-500 text-xs">No match for <span class="text-brand-400 font-mono">"${esc(query)}"</span></div>`;
        return;
      }
      cmdResults.innerHTML = filtered.map((item) => `
        <button type="button" data-i="${searchCatalog.indexOf(item)}" class="cmd-item w-full text-left flex items-center justify-between p-2.5 rounded-lg hover:bg-brand-500/20 text-slate-300 hover:text-white transition">
          <span>${esc(item.title)}</span>
          <span class="text-[10px] text-brand-400 px-1.5 py-0.5 rounded bg-brand-500/10 border border-brand-500/20">${esc(item.category)}</span>
        </button>`).join('');
      cmdResults.querySelectorAll('.cmd-item').forEach((b) => b.addEventListener('click', () => goToResult(searchCatalog[Number(b.dataset.i)])));
    }
    cmdInput.addEventListener('input', (e) => filterCmdResults(e.target.value));
    cmdInput.addEventListener('keydown', (e) => {
      const items = cmdResults.querySelectorAll('.cmd-item');
      if (!items.length) return;
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        activeCmdIndex = e.key === 'ArrowDown' ? (activeCmdIndex + 1) % items.length : (activeCmdIndex - 1 + items.length) % items.length;
        items.forEach((it, idx) => it.classList.toggle('bg-brand-500/30', idx === activeCmdIndex));
        items[activeCmdIndex].scrollIntoView({ block: 'nearest' });
      } else if (e.key === 'Enter') {
        e.preventDefault();
        items[Math.max(0, activeCmdIndex)].click();
      }
    });
    function openCmdModal() {
      cmdModal.classList.remove('hidden');
      cmdInput.value = '';
      filterCmdResults('');
      beginTrap([cmdModal], cmdInput);
    }
    function closeCmdModal(restore = true) {
      if (cmdModal.classList.contains('hidden')) return;
      cmdModal.classList.add('hidden');
      endTrap(restore);
    }
    $('cmdBtn').addEventListener('click', openCmdModal);
    cmdModal.addEventListener('click', (e) => { if (!e.target.closest('.glass-card')) closeCmdModal(); });

    const menuBtn = $('menuBtn');
    const mobileNav = $('mobileNav');
    function setMenuOpen(open, restore = true) {
      const was = menuBtn.getAttribute('aria-expanded') === 'true';
      mobileNav.classList.toggle('hidden', !open);
      menuBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
      menuBtn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
      $('menuIconOpen').classList.toggle('hidden', open);
      $('menuIconClose').classList.toggle('hidden', !open);
      if (open && !was) beginTrap([menuBtn, mobileNav], mobileNav.querySelector('a'));
      if (!open && was) endTrap(restore);
    }
    menuBtn.addEventListener('click', () => setMenuOpen(menuBtn.getAttribute('aria-expanded') !== 'true'));
    mobileNav.querySelectorAll('a').forEach((link) => link.addEventListener('click', () => setMenuOpen(false)));
    $('mobileCargo').addEventListener('click', () => {
      copyCommand('go get github.com/bharathvbcr/ojas/go');
      setMenuOpen(false);
    });

    window.addEventListener('keydown', (e) => {
      const typing = ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName);
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        openCmdModal();
      } else if (e.key === '/' && !typing) {
        e.preventDefault();
        openCmdModal();
      } else if (e.key === 'Escape') {
        const dialog = !cmdModal.classList.contains('hidden');
        const menu = menuBtn.getAttribute('aria-expanded') === 'true';
        if (!dialog && !menu) return;
        closeCmdModal(false);
        setMenuOpen(false, false);
        endTrap(true);
      }
    });

    const announcement = $('announcement');
    let announceDismissed = false;
    try { announceDismissed = sessionStorage.getItem('ojas-announce-dismissed') === '1'; } catch (err) { announceDismissed = false; }
    if (announceDismissed) {
      announcement.remove();
    } else {
      $('dismissAnnounce').addEventListener('click', () => {
        announcement.remove();
        try { sessionStorage.setItem('ojas-announce-dismissed', '1'); } catch (err) { /* private mode */ }
      });
    }

    function updateScrollSpy() {
      const marker = window.scrollY + 130;
      let currentId = '';
      const sections = document.querySelectorAll('main section[id]');
      sections.forEach((sec) => {
        if (sec.offsetTop <= marker) currentId = sec.id;
      });
      const root = document.documentElement;
      if (sections.length && window.scrollY >= root.scrollHeight - root.clientHeight - 2) {
        currentId = sections[sections.length - 1].id;
      }
      document.querySelectorAll('[data-spy]').forEach((link) => {
        if (link.getAttribute('href') === '#' + currentId) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
      let docId = '';
      if (currentId === 'docs') {
        document.querySelectorAll('.doc-article').forEach((art) => {
          if (art.getBoundingClientRect().top + window.scrollY <= marker) docId = art.id;
        });
      }
      document.querySelectorAll('[data-doc-spy]').forEach((link) => {
        if (link.getAttribute('href') === '#' + docId) link.setAttribute('aria-current', 'true');
        else link.removeAttribute('aria-current');
      });
      const doc = document.documentElement;
      const max = doc.scrollHeight - doc.clientHeight;
      const pct = max > 0 ? Math.min(100, Math.max(0, (window.scrollY / max) * 100)) : 0;
      $('readProgressBar').style.width = pct + '%';
      $('readProgress').setAttribute('aria-valuenow', String(Math.round(pct)));
    }
    function revealHash() {
      const id = decodeURIComponent((location.hash || '').replace(/^#/, ''));
      if (!id) return;
      const el = document.getElementById(id);
      if (!el || !el.closest('main')) return;
      el.scrollIntoView();
      updateScrollSpy();
    }
    updateScrollSpy();
    revealHash();
    window.addEventListener('scroll', updateScrollSpy, { passive: true });
    window.addEventListener('hashchange', revealHash);
    window.addEventListener('resize', () => {
      if (window.matchMedia('(min-width: 1280px)').matches) setMenuOpen(false);
    });

    let toastTimeout;
    function showToast(msg) {
      const toast = $('toast');
      $('toastMsg').textContent = msg;
      toast.classList.remove('translate-y-20', 'opacity-0', 'pointer-events-none');
      clearTimeout(toastTimeout);
      toastTimeout = setTimeout(() => {
        toast.classList.add('translate-y-20', 'opacity-0', 'pointer-events-none');
      }, 3000);
    }
