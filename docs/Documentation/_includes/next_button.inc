.. "Next" button linking to the next page of the documentation.
.. It can be added to any page, at the bottom, with:
..
..     .. include:: /_includes/next_button.inc
..
.. (the leading "/" makes the path relative to the docs root, so the same line
.. works from any folder). Pages that do not include this file get no button.
..
.. The link is filled in when the page loads, from the next page in the
.. toctree, so reordering an index needs no changes here. The button is hidden
.. when there is no next page or when the next page is a section index.

.. raw:: html

    <div style="display: flex; justify-content: flex-end; margin-top: 3em; padding-top: 1.5em; border-top: 1px solid #ddd;">
      <a id="next-page-button" href="#"
         style="display: inline-block; padding: 0.6em 1.2em; border: 1px solid #ccc; border-radius: 6px; background: #f8f8f8; color: #004b6b; font-weight: bold; text-decoration: none; border-bottom: 1px solid #ccc;">
        Next &rarr;
      </a>
    </div>
    <script>
      // Point the button at the next page in the toctree, which Sphinx writes
      // into the page head as <link rel="next">.
      (function () {
        var button = document.getElementById("next-page-button");
        var next = document.querySelector('link[rel="next"]');
        var href = next ? next.getAttribute("href") : null;
        if (!href || /(^|\/)index\.html$/.test(href)) {
          button.parentElement.style.display = "none";
          return;
        }
        button.href = href;
      })();
    </script>
