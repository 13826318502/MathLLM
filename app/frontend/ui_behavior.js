(() => {
  const bindShortcuts = () => {
    const input = document.querySelector("#question-box textarea");
    const submit = document.querySelector("#submit-button button");
    if (!input || !submit || input.dataset.mathllmBound === "true") return;

    input.dataset.mathllmBound = "true";
    input.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        submit.click();
      }
    });
  };

  bindShortcuts();
  new MutationObserver(bindShortcuts).observe(document.body, {
    childList: true,
    subtree: true,
  });
})();
