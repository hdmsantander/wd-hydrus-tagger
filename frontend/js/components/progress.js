/**
 * Progress bar component.
 */

import { $ } from '../utils/dom.js';

export function showProgress(total) {
    $('#progress-overlay').style.display = 'flex';
    $('#progress-bar').style.width = '0%';
    $('#progress-text').textContent = `0 / ${total}`;
    $('#progress-title').textContent = '處理中...';
}

export function updateProgress(current, total, message = null) {
    const pct = total > 0 ? (current / total * 100) : 0;
    $('#progress-bar').style.width = `${pct}%`;
    $('#progress-text').textContent = `${current} / ${total}`;
    if (message) {
        $('#progress-title').textContent = message;
    }
}

export function hideProgress() {
    setTimeout(() => {
        $('#progress-overlay').style.display = 'none';
    }, 500);
}
