(function () {
  // Copy buttons on code blocks.
  document.querySelectorAll('.code .copy').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var code = btn.parentElement.querySelector('code');
      var done = function () { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy'; }, 1500); };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(code.textContent).then(done, function () {});
      }
    });
  });

  // Mermaid diagrams: loaded from cdnjs only on pages that have one. The pinned
  // version carries its SRI hash; if the script fails, the source text stays visible.
  var diagrams = document.querySelectorAll('pre.mermaid');
  if (diagrams.length) {
    var s = document.createElement('script');
    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/mermaid/11.15.0/mermaid.min.js';
    s.integrity = 'sha512-HH52omhHpZF6RfVnGiQwYgYm4H/ya2xsZYLl5xJ4+tLfX+rN4+8zF7V/H/KLeicPrKZYi1g6iBmVkk2AhXTGlg==';
    s.crossOrigin = 'anonymous';
    s.referrerPolicy = 'no-referrer';
    s.onload = function () {
      if (!window.mermaid) return;
      window.mermaid.initialize({ startOnLoad: false, theme: 'dark', securityLevel: 'strict', fontFamily: 'Plus Jakarta Sans, system-ui, sans-serif' });
      window.mermaid.run({ nodes: Array.prototype.slice.call(diagrams) }).then(function () {
        document.querySelectorAll('figure.diagram').forEach(function (f) { f.classList.add('drawn'); });
      }).catch(function () {});
    };
    document.head.appendChild(s);
  }

  // Sidebar: open on wide screens, collapsed on phones.
  var side = document.querySelector('.ref-side details');
  if (side && window.matchMedia('(max-width: 1023px)').matches) side.removeAttribute('open');

  // On-page TOC highlight.
  var links = Array.prototype.slice.call(document.querySelectorAll('.ref-toc a'));
  if (links.length && 'IntersectionObserver' in window) {
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          links.forEach(function (a) { a.classList.remove('active'); });
          if (byId[e.target.id]) byId[e.target.id].classList.add('active');
        }
      });
    }, { rootMargin: '-80px 0px -70% 0px' });
    Object.keys(byId).forEach(function (id) { var el = document.getElementById(id); if (el) obs.observe(el); });
  }

  // Index filter over title, description and section headings.
  var input = document.getElementById('refFilter');
  var data = document.getElementById('refIndex');
  if (input && data) {
    var entries = JSON.parse(data.textContent);
    var count = document.getElementById('refCount');
    var empty = document.getElementById('refEmpty');
    var items = {};
    document.querySelectorAll('[data-slug]').forEach(function (li) { items[li.dataset.slug] = li; });
    var apply = function () {
      var q = input.value.trim().toLowerCase();
      var shown = 0;
      entries.forEach(function (e) {
        var hay = (e.title + ' ' + e.desc + ' ' + e.heads.join(' ')).toLowerCase();
        var ok = !q || q.split(/\s+/).every(function (t) { return hay.indexOf(t) !== -1; });
        items[e.slug].hidden = !ok;
        if (ok) shown++;
      });
      document.querySelectorAll('.ref-cat').forEach(function (s) {
        s.hidden = !s.querySelector('[data-slug]:not([hidden])');
      });
      count.textContent = q ? shown + ' of ' + entries.length + ' pages match' : entries.length + ' pages';
      empty.classList.toggle('hidden', shown !== 0);
    };
    input.addEventListener('input', apply);
    apply();
  }
})();
