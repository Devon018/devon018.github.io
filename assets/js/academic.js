(() => {
  "use strict";

  const navigation = document.querySelector(".academic-nav");
  const toggle = navigation?.querySelector(".academic-nav__toggle");
  const menu = navigation?.querySelector(".academic-nav__links");
  if (!navigation || !toggle || !menu) return;

  navigation.setAttribute("data-enhanced", "");
  toggle.hidden = false;

  const setOpen = (open) => {
    navigation.toggleAttribute("data-open", open);
    toggle.setAttribute("aria-expanded", String(open));
    toggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
  };

  toggle.addEventListener("click", () => {
    setOpen(toggle.getAttribute("aria-expanded") !== "true");
  });
  menu.addEventListener("click", (event) => {
    if (event.target.closest("a")) setOpen(false);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && toggle.getAttribute("aria-expanded") === "true") {
      setOpen(false);
      toggle.focus();
    }
  });

  const sectionLinks = [...menu.querySelectorAll("a")].filter((link) => {
    const url = new URL(link.href);
    return url.pathname === location.pathname && url.hash;
  });
  if (!("IntersectionObserver" in window)) return;
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const link of sectionLinks) {
        if (new URL(link.href).hash === `#${entry.target.id}`) {
          link.setAttribute("aria-current", "location");
        } else {
          link.removeAttribute("aria-current");
        }
      }
    }
  }, { rootMargin: "-10% 0px -70% 0px" });
  for (const link of sectionLinks) {
    const section = document.getElementById(new URL(link.href).hash.slice(1));
    if (section) observer.observe(section);
  }
})();
