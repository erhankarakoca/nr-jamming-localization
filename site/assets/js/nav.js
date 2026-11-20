export function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

export function reportError(target, error) {
  const message = element("p", error.message, "error");
  message.setAttribute("role", "alert");
  target.replaceChildren(message);
  target.setAttribute("aria-busy", "false");
}

export function options(select, entries) {
  select.replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
  select.disabled = false;
}

document.querySelector(".skip-link")?.addEventListener("click", () => {
  document.querySelector("main")?.focus();
});
