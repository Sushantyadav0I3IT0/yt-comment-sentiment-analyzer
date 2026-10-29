(() => {
  const BATCH_SIZE = 40;
  const COMMENT_SELECTOR = 'ytd-comment-view-model, ytd-comment-renderer';
  const TEXT_SELECTOR = '#content-text, .comment-text';
  const LABELS = {
    '-1': { text: 'Negative', className: 'negative' },
    '0': { text: 'Neutral', className: 'neutral' },
    '1': { text: 'Positive', className: 'positive' },
  };

  if (document.getElementById('yt-sentiment-panel')) {
    return;
  }

  const panel = document.createElement('section');
  panel.id = 'yt-sentiment-panel';
  const heading = document.createElement('div');
  heading.className = 'yt-sentiment-heading';
  heading.textContent = 'Comment sentiment';
  const button = document.createElement('button');
  button.id = 'yt-sentiment-analyze';
  button.type = 'button';
  button.textContent = 'Analyze loaded comments';
  const status = document.createElement('div');
  status.id = 'yt-sentiment-status';
  status.setAttribute('role', 'status');
  status.textContent = 'Uses your local API at 127.0.0.1:5058.';
  const summary = document.createElement('div');
  summary.id = 'yt-sentiment-summary';
  summary.setAttribute('aria-live', 'polite');
  panel.append(heading, button, status, summary);
  document.body.append(panel);

  async function analyzeBatch(comments) {
    const result = await chrome.runtime.sendMessage({
      type: 'analyze-comments',
      comments,
    });
    if (!result?.ok) {
      throw new Error(result?.error || 'Could not reach the local sentiment API.');
    }
    if (
      !Array.isArray(result.predictions)
      || result.predictions.length !== comments.length
    ) {
      throw new Error('The API returned an unexpected prediction response.');
    }
    return result.predictions;
  }

  function getLoadedComments() {
    const seenTextElements = new Set();
    return [...document.querySelectorAll(COMMENT_SELECTOR)]
      .map((commentElement) => ({
        commentElement,
        textElement: commentElement.querySelector(TEXT_SELECTOR),
      }))
      .filter(({ textElement }) => {
        if (
          !textElement
          || !textElement.textContent.trim()
          || seenTextElements.has(textElement)
        ) {
          return false;
        }
        seenTextElements.add(textElement);
        return true;
      });
  }

  function showPrediction(textElement, value) {
    const label = LABELS[String(value)];
    if (!label) {
      throw new Error(`Unknown sentiment label returned: ${value}`);
    }

    const parent = textElement.parentElement;
    if (!parent) {
      return;
    }

    let badge = parent.querySelector(
      ':scope > .yt-sentiment-badge'
    );
    if (!badge) {
      badge = document.createElement('span');
      badge.className = 'yt-sentiment-badge';
      textElement.insertAdjacentElement('afterend', badge);
    }
    badge.className = `yt-sentiment-badge ${label.className}`;
    badge.textContent = label.text;
  }

  button.addEventListener('click', async () => {
    const comments = getLoadedComments();
    if (!comments.length) {
      status.textContent = 'No loaded comments found. Open a video and scroll to its comments.';
      return;
    }

    button.disabled = true;
    let analyzedCount = 0;
    const counts = { '-1': 0, '0': 0, '1': 0 };
    summary.textContent = '';
    status.textContent = `Analyzing ${comments.length} loaded comments...`;

    try {
      for (let start = 0; start < comments.length; start += BATCH_SIZE) {
        const batch = comments.slice(start, start + BATCH_SIZE);
        const predictions = await analyzeBatch(
          batch.map(({ textElement }) => textElement.textContent.trim())
        );

        predictions.forEach((prediction, index) => {
          const sentiment = String(prediction.sentiment);
          if (!Object.hasOwn(counts, sentiment)) {
            throw new Error(`Unknown sentiment label returned: ${sentiment}`);
          }
          counts[sentiment] += 1;
          showPrediction(batch[index].textElement, sentiment);
        });
        analyzedCount += batch.length;
        status.textContent = `Analyzed ${analyzedCount} of ${comments.length} loaded comments.`;
        const percentage = (count) =>
          ((count / analyzedCount) * 100).toFixed(1);
        summary.textContent =
          `Positive: ${counts['1']} (${percentage(counts['1'])}%) · `
          + `Neutral: ${counts['0']} (${percentage(counts['0'])}%) · `
          + `Negative: ${counts['-1']} (${percentage(counts['-1'])}%)`;
      }
      status.textContent =
        `Done — percentages are out of ${analyzedCount} analyzed comments. Scroll to see each label.`;
    } catch (error) {
      status.textContent = `${error.message} Confirm the API is running at http://127.0.0.1:5058/health.`;
    } finally {
      button.disabled = false;
    }
  });
})();
