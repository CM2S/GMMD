.. "Previous" button linking to the previous page of the documentation.
.. It can be added to any page, at the bottom, with:
..
..     .. include:: /_includes/previous_button.inc
..
.. (the leading "/" makes the path relative to the docs root, so the same line
.. works from any folder). Pages that do not include this file get no button.
..
.. The link is filled in when the page loads, from the previous page in the
.. toctree, so reordering an index needs no changes here. The button is hidden
.. when there is no previous page or when the previous page is a section index.
.. If the page also includes next_button.inc (after this file), both buttons
.. are shown on the same row (previous on the left, next on the right).

.. raw:: html

    <div id="previous-page-container" style="display: flex; justify-content: flex-start; margin-top: 3em; padding-top: 1.5em; border-top: 1px solid #ddd;">
      <a id="previous-page-button" href="#"
         style="display: inline-block; padding: 0.6em 1.2em; border: 1px solid #ccc; border-radius: 6px; background: #f8f8f8; color: #004b6b; font-weight: bold; text-decoration: none; border-bottom: 1px solid #ccc;">
        &larr; Previous
      </a>
    </div>
    <script>
      // Point the button at the previous page in the toctree, which Sphinx
      // writes into the page head as <link rel="prev">.
      (function () {
        var container = document.getElementById("previous-page-container");
        var button = document.getElementById("previous-page-button");
        var prev = document.querySelector('link[rel="prev"]');
        var href = prev ? prev.getAttribute("href") : null;
        if (!href || /(^|\/)index\.html$/.test(href)) {
          container.remove();
          return;
        }
        button.href = href;
        // If there is also a "Next" button, put both on the same row.
        document.addEventListener("DOMContentLoaded", function () {
          var nextButton = document.getElementById("next-page-button");
          if (nextButton && nextButton.parentElement.style.display !== "none") {
            var nextContainer = nextButton.parentElement;
            nextContainer.style.justifyContent = "space-between";
            nextContainer.insertBefore(button, nextButton);
            container.remove();
          }
        });
      })();
    </script>
