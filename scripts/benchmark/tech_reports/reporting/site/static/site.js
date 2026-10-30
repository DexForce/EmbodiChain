"use strict";

// Keep the reading column visible on laptops and phones; wide screens use a side TOC.
document.querySelectorAll(".toc-panel details").forEach((details) => {
  details.open = window.matchMedia("(min-width: 1500px)").matches;
});

const menuButton = document.querySelector(".nav-toggle");
menuButton.addEventListener("click", () => {
  const expanded = menuButton.getAttribute("aria-expanded") === "true";
  menuButton.setAttribute("aria-expanded", String(!expanded));
  document.querySelector(".sidebar").classList.toggle("is-open", !expanded);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    menuButton.setAttribute("aria-expanded", "false");
    document.querySelector(".sidebar").classList.remove("is-open");
  }
});
document.querySelectorAll("[data-tex]").forEach((element) => {
  if (window.katex) {
    katex.render(element.dataset.tex, element, {
      displayMode: element.classList.contains("math-display"),
      throwOnError: false,
      strict: "ignore",
      trust: false,
    });
  }
});
