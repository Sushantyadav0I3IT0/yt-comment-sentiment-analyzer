const API_URL = 'http://127.0.0.1:5058';

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== 'analyze-comments' || !Array.isArray(message.comments)) {
    return false;
  }
  if (
    message.comments.length > 40
    || message.comments.some((comment) => typeof comment !== 'string')
  ) {
    sendResponse({ ok: false, error: 'Invalid comment batch.' });
    return false;
  }

  fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ comments: message.comments }),
  })
    .then(async (response) => {
      const body = await response.json();
      if (!response.ok) {
        throw new Error(body.error || `API returned HTTP ${response.status}`);
      }
      sendResponse({ ok: true, predictions: body });
    })
    .catch((error) => {
      sendResponse({ ok: false, error: error.message });
    });

  return true;
});
