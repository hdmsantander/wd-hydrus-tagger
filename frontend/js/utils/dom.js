/**
 * DOM helper utilities.
 */

export function $(selector) {
    return document.querySelector(selector);
}

export function $$(selector) {
    return document.querySelectorAll(selector);
}

export function el(tag, attrs = {}, children = []) {
    const elem = document.createElement(tag);
    for (const [key, val] of Object.entries(attrs)) {
        if (key === 'className') elem.className = val;
        else if (key === 'textContent') elem.textContent = val;
        else if (key === 'innerHTML') elem.innerHTML = val;
        else if (key.startsWith('on')) elem.addEventListener(key.slice(2).toLowerCase(), val);
        else elem.setAttribute(key, val);
    }
    for (const child of children) {
        if (typeof child === 'string') elem.appendChild(document.createTextNode(child));
        else if (child) elem.appendChild(child);
    }
    return elem;
}

export function show(elem) {
    if (typeof elem === 'string') elem = $(elem);
    if (elem) elem.style.display = '';
}

export function hide(elem) {
    if (typeof elem === 'string') elem = $(elem);
    if (elem) elem.style.display = 'none';
}
