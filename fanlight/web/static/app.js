/* Fanlight web app. No framework, no build step. */
"use strict";

const SIGNAL_NAMES = {
  category: "Category",
  tags: "Topic tags",
  content: "Similar text",
  freshness: "Freshness",
  source: "Outlet",
  popularity: "Coverage",
};
const MIX_COLOURS = ["#D9B46A", "#C58BEA", "#8E6BBF", "#E3CFA1", "#6B4B94", "#F0B5D0", "#A78A55", "#5B3C80"];
const HALF_LIFE = { min: 6, max: 24 * 30 }; // hours, log scale on the slider
const STORAGE_KEY = "fanlight:v1";
const MY_READERS_KEY = "fanlight:readers";
const META_POLL_MS = 2 * 60 * 1000;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = {
  meta: null,
  readers: [],
  readerId: null,
  category: null,
  tuning: {},
  feed: null,
  seen: new Set(),
  pendingViews: [],
  request: 0,
};

const $ = (selector, root = document) => root.querySelector(selector);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
};

/* ------------------------------------------------------------------ */
/* Storage (best effort: the preview iframe may forbid it)            */
/* ------------------------------------------------------------------ */

function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch { return {}; }
}
/* Readers created in this browser. In public mode these ids are the only way
   back to them, because the server will not list anybody's readers. */
function myReaders() {
  try { return JSON.parse(localStorage.getItem(MY_READERS_KEY)) || []; } catch { return []; }
}
function rememberReader(id, keep = true) {
  try {
    const ids = myReaders().filter((x) => x !== id);
    if (keep) ids.unshift(id);
    localStorage.setItem(MY_READERS_KEY, JSON.stringify(ids.slice(0, 20)));
  } catch { /* storage unavailable */ }
}
function savePrefs() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ readerId: state.readerId, tuning: state.tuning }));
  } catch { /* storage unavailable */ }
}

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      if (data.detail) detail = typeof data.detail === "string" ? data.detail : data.detail.map((d) => d.msg).join("; ");
    } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

/* ------------------------------------------------------------------ */
/* Formatting                                                          */
/* ------------------------------------------------------------------ */

const isHindi = (text) => /[ऀ-ॿ]/.test(text || "");

function age(hours) {
  if (hours < 1) return "under an hour old";
  if (hours < 24) return `${Math.round(hours)} h old`;  // matches the server's "Published N hours ago"
  const days = Math.round(hours / 24);
  return days === 1 ? "1 day old" : `${days} days old`;
}

function hoursLabel(hours) {
  if (hours < 48) return `${Math.round(hours)} h`;
  return `${Math.round(hours / 24)} days`;
}

function editionDate(iso) {
  return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
}

/* ------------------------------------------------------------------ */
/* Masthead fan                                                        */
/* ------------------------------------------------------------------ */

function drawFan() {
  // A sunburst rising behind the wordmark. Rays and arcs stop short of the
  // horizon (FROM..TO degrees) so they frame the letters instead of crossing them.
  const svg = $(".fan");
  const group = $(".rays", svg);
  const NS = "http://www.w3.org/2000/svg";
  const cx = 300, cy = 170, inner = 96, outer = 166, count = 15, FROM = 22, TO = 158;
  const point = (radius, degrees) => {
    const a = (degrees * Math.PI) / 180;
    return [cx + radius * Math.cos(a), cy - radius * Math.sin(a)].map((v) => v.toFixed(1));
  };
  for (let r = 0; r < count; r++) {
    const degrees = TO - ((TO - FROM) * r) / (count - 1);
    const length = r % 2 === 0 ? outer : outer - 32;
    const [x1, y1] = point(inner, degrees);
    const [x2, y2] = point(length, degrees);
    const line = document.createElementNS(NS, "line");
    line.setAttribute("x1", x1); line.setAttribute("y1", y1);
    line.setAttribute("x2", x2); line.setAttribute("y2", y2);
    line.style.setProperty("--len", (length - inner).toFixed(0));
    line.style.setProperty("--delay", `${Math.abs(r - (count - 1) / 2) * 50}ms`);
    group.append(line);
  }
  for (const [radius, extra] of [[150, ""], [88, " arc-inner"]]) {
    const [x1, y1] = point(radius, TO + 4);
    const [x2, y2] = point(radius, FROM - 4);
    const arc = document.createElementNS(NS, "path");
    arc.setAttribute("d", `M${x1} ${y1} A${radius} ${radius} 0 0 1 ${x2} ${y2}`);
    arc.setAttribute("class", `arc${extra}`);
    svg.append(arc);
  }
  svg.classList.add("drawing");
}

/* ------------------------------------------------------------------ */
/* Tuning dials                                                        */
/* ------------------------------------------------------------------ */

const toSlider = (hours) => Math.round((100 * Math.log(hours / HALF_LIFE.min)) / Math.log(HALF_LIFE.max / HALF_LIFE.min));
const fromSlider = (value) => HALF_LIFE.min * Math.pow(HALF_LIFE.max / HALF_LIFE.min, value / 100);

function effectiveTuning() {
  const defaults = state.meta.defaults;
  return {
    diversity: state.tuning.diversity ?? defaults.diversity,
    discovery: state.tuning.discovery ?? defaults.discovery,
    half_life_hours: state.tuning.half_life_hours ?? defaults.half_life_hours,
    hide_read: state.tuning.hide_read ?? defaults.hide_read,
    weights: { ...defaults.weights, ...(state.tuning.weights || {}) },
  };
}

function paintRange(input) {
  const min = Number(input.min), max = Number(input.max);
  input.style.setProperty("--fill", `${((Number(input.value) - min) / (max - min)) * 100}%`);
}

function syncDials() {
  const t = effectiveTuning();
  const set = (id, value, label) => {
    const input = $(`#dial-${id}`);
    input.value = value;
    paintRange(input);
    $(`#out-${id}`).textContent = label;
  };
  set("diversity", t.diversity, `${Math.round(t.diversity * 100)}%`);
  set("discovery", t.discovery, t.discovery === 0 ? "off" : `1 in ${Math.round(1 / t.discovery)}`);
  set("halflife", toSlider(t.half_life_hours), `halves every ${hoursLabel(t.half_life_hours)}`);
  $("#dial-hideread").checked = t.hide_read;
  for (const [signal, value] of Object.entries(t.weights)) {
    const input = $(`#w-${signal}`);
    if (!input) continue;
    input.value = value;
    paintRange(input);
    $(`#out-w-${signal}`).textContent = value.toFixed(2);
  }
}

function buildDials() {
  const holder = $("#weight-dials");
  for (const signal of state.meta.signals) {
    holder.append(
      el("div", { class: "dial" },
        el("label", { for: `w-${signal.key}` }, SIGNAL_NAMES[signal.key] || signal.key, el("output", { id: `out-w-${signal.key}` })),
        el("input", { type: "range", id: `w-${signal.key}`, min: "0", max: "1", step: "0.02", "data-signal": signal.key }),
        el("p", { class: "hint", text: signal.help }),
      ),
    );
  }
  let timer = null;
  const changed = () => {
    syncDials();
    savePrefs();
    clearTimeout(timer);
    timer = setTimeout(() => loadFeed({ animate: true }), 220);
  };
  $("#dial-diversity").addEventListener("input", (e) => { state.tuning.diversity = Number(e.target.value); changed(); });
  $("#dial-discovery").addEventListener("input", (e) => { state.tuning.discovery = Number(e.target.value); changed(); });
  $("#dial-halflife").addEventListener("input", (e) => {
    state.tuning.half_life_hours = Math.round(fromSlider(Number(e.target.value)));
    changed();
  });
  $("#dial-hideread").addEventListener("change", (e) => { state.tuning.hide_read = e.target.checked; changed(); });
  holder.addEventListener("input", (e) => {
    const signal = e.target.dataset.signal;
    if (!signal) return;
    state.tuning.weights = { ...(state.tuning.weights || {}), [signal]: Number(e.target.value) };
    changed();
  });
  $("#reset-tuning").addEventListener("click", () => { state.tuning = {}; changed(); });
  syncDials();
}

/* ------------------------------------------------------------------ */
/* Readers                                                             */
/* ------------------------------------------------------------------ */

function renderReaderSelect() {
  const select = $("#reader-select");
  select.replaceChildren();
  const created = state.readers.filter((r) => r.kind === "reader");
  const imported = state.readers.filter((r) => r.kind !== "reader");
  const option = (r) => el("option", { value: r.id, text: r.follows.length ? `${r.name}, ${r.follows.slice(0, 2).join(" and ")}` : `${r.name}, no topics yet` });
  if (created.length) select.append(el("optgroup", { label: "Your readers" }, created.map(option)));
  if (imported.length) select.append(el("optgroup", { label: "Readers from the dataset" }, imported.map(option)));
  select.value = state.readerId;
}

function renderReader(reader) {
  const card = $("#reader-card");
  const cats = reader.interests.categories;
  const actions = reader.activity;
  const total = actions.open + actions.like + actions.save + actions.hide;
  const parts = [
    el("p", { class: "cohort" }, "Cohort: ", el("b", { text: reader.cohort })),
  ];
  if (cats.length) {
    parts.push(
      el("ul", { class: "affinity", "aria-label": "What the ranker thinks this reader likes" },
        cats.slice(0, 6).map((c) =>
          el("li", {},
            el("span", {}, c.name, c.stated ? null : el("span", { class: "learned", text: "learned" })),
            el("span", { class: "bar", title: `${Math.round(c.affinity * 100)}%` }, el("i", { style: `width:${Math.max(4, c.affinity * 100)}%` })),
          ),
        ),
      ),
    );
  } else {
    parts.push(el("p", { class: "activity", text: "No interests yet. Open or like a few stories and the feed will start to follow you." }));
  }
  if (reader.interests.avoided.length) {
    parts.push(el("p", { class: "avoided", text: `Showing less of ${reader.interests.avoided.map((a) => a.name).join(", ")}` }));
  }
  let history = "No reading history yet.";
  if (total) history = `${actions.open} opened, ${actions.like} liked, ${actions.save} saved, ${actions.hide} hidden`;
  else if (actions.view) history = `Nothing opened yet, ${actions.view} stories scrolled past.`;
  parts.push(el("p", { class: "activity", text: history }));
  const controls = el("div", { class: "reader-actions" });
  if (total || actions.view) {
    controls.append(el("button", { type: "button", class: "link-button", text: "Clear reading history", onclick: clearHistory }));
  }
  if (reader.editable) {
    controls.append(el("button", { type: "button", class: "link-button", text: "Delete reader", onclick: deleteReader }));
  }
  if (controls.childElementCount) parts.push(controls);
  card.replaceChildren(...parts);
}

async function clearHistory() {
  await api(`/api/readers/${state.readerId}/history`, { method: "DELETE" });
  state.seen.clear();
  toast("Reading history cleared");
  loadFeed({ animate: true });
}

async function deleteReader() {
  const name = state.feed?.reader?.name || "this reader";
  if (!window.confirm(`Delete ${name} and their history?`)) return;
  await api(`/api/readers/${state.readerId}`, { method: "DELETE" });
  rememberReader(state.readerId, false);
  await refreshReaders();
  toast(`Deleted ${name}`);
  if (state.readers.length) selectReader(state.readers[0].id);
  else showWelcome();
}

async function refreshReaders() {
  const mine = myReaders();
  const own = mine.length ? await api(`/api/readers?ids=${encodeURIComponent(mine.join(","))}`) : [];
  const listed = state.meta?.public ? [] : await api("/api/readers?limit=80");
  const ownIds = new Set(own.map((r) => r.id));
  state.readers = [...own, ...listed.filter((r) => !ownIds.has(r.id))];
}

function showWelcome() {
  state.readerId = null;
  state.feed = null;
  renderReaderSelect();
  $("#reader-card").replaceChildren();
  $("#edition").textContent = editionLine();
  $("#shape").replaceChildren(el("p", { class: "shape-note", text: "Create a reader to see the shape of their feed." }));
  $("#feed-foot").textContent = "";
  $("#feed-list").replaceChildren(
    el("li", { class: "empty welcome" },
      el("h3", { text: "Make your feed" }),
      el("p", { text: "Pick a language and a few topics. Fanlight ranks today's stories for you and learns from what you open, like, save or skip." }),
      el("button", { type: "button", class: "button", text: "Pick my topics", onclick: () => openReaderDialog(true) }),
    ),
  );
}

function selectReader(id) {
  state.readerId = id;
  state.seen.clear();
  savePrefs();
  renderReaderSelect();
  loadFeed({ animate: false });
}

function openReaderDialog(welcome) {
  const form = $("#reader-form");
  form.reset();
  $("#dialog-error").textContent = "";
  $("#dialog-title").textContent = welcome ? "Make your feed" : "New reader";
  $("#dialog-submit").textContent = welcome ? "Show my feed" : "Create reader";
  $("#reader-dialog").showModal();
}

function setupDialog() {
  const dialog = $("#reader-dialog");
  const form = $("#reader-form");
  const chip = (group, value, label, pressed = false) =>
    el("button", { type: "button", class: "chip", "data-group": group, "data-value": value, "aria-pressed": String(pressed), text: label, onclick: (e) => {
      const b = e.currentTarget;
      b.setAttribute("aria-pressed", String(b.getAttribute("aria-pressed") !== "true"));
    } });
  $("#dialog-languages").append(...state.meta.languages.map((l, i) => chip("language", l, l[0].toUpperCase() + l.slice(1), i === 0)));
  $("#dialog-categories").append(...state.meta.categories.slice(0, 30).map((c) => chip("category", c.key, c.name)));

  $("#new-reader").addEventListener("click", () => openReaderDialog(false));
  $("#dialog-cancel").addEventListener("click", () => dialog.close());
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const welcome = $("#dialog-title").textContent === "Make your feed";
    const picked = (group) => [...form.querySelectorAll(`[data-group="${group}"][aria-pressed="true"]`)].map((b) => b.dataset.value);
    const languages = picked("language");
    if (!languages.length) { $("#dialog-error").textContent = "Pick at least one language."; return; }
    try {
      const reader = await api("/api/readers", { method: "POST", body: { name: form.name.value.trim() || "You", languages, categories: picked("category") } });
      dialog.close();
      rememberReader(reader.id);
      await refreshReaders();
      selectReader(reader.id);
      toast(welcome ? "Your feed is ready." : `Created ${reader.name}`);
    } catch (error) {
      $("#dialog-error").textContent = error.message;
    }
  });
}

/* ------------------------------------------------------------------ */
/* Category filters                                                    */
/* ------------------------------------------------------------------ */

function renderFilters() {
  const track = $("#filters");
  const button = (key, label) =>
    el("button", { type: "button", class: "chip", "aria-pressed": String(state.category === key), text: label, onclick: () => {
      state.category = key;
      renderFilters();
      loadFeed({ animate: false });
    } });
  track.replaceChildren(button(null, "All stories"), ...state.meta.categories.slice(0, 16).map((c) => button(c.key, c.name)));
}

/* ------------------------------------------------------------------ */
/* Feed                                                                */
/* ------------------------------------------------------------------ */

function feedUrl() {
  const t = effectiveTuning();
  const params = new URLSearchParams({
    k: "20",
    diversity: t.diversity,
    discovery: t.discovery,
    half_life_hours: t.half_life_hours,
    hide_read: t.hide_read,
  });
  for (const [signal, value] of Object.entries(state.tuning.weights || {})) params.set(`w_${signal}`, value);
  if (state.category) params.set("category", state.category);
  return `/api/readers/${encodeURIComponent(state.readerId)}/feed?${params}`;
}

async function loadFeed({ animate = true } = {}) {
  const ticket = ++state.request;
  const list = $("#feed-list");
  list.setAttribute("aria-busy", "true");
  if (!state.feed) list.replaceChildren(...Array.from({ length: 5 }, () => el("li", { class: "ghost-story", "aria-hidden": "true" })));
  let feed;
  try {
    feed = await api(feedUrl());
  } catch (error) {
    if (ticket !== state.request) return;
    list.replaceChildren(el("li", { class: "error-box" }, `The feed did not load: ${error.message}. Check that the server is running, then reload the page.`));
    list.setAttribute("aria-busy", "false");
    return;
  }
  if (ticket !== state.request) return; // a newer request superseded this one
  state.feed = feed;
  renderReader(feed.reader);
  renderEdition(feed.reader);
  renderFeed(feed, animate && !reduceMotion);
  renderShape(feed);
  list.setAttribute("aria-busy", "false");
}

function minutesAgo(iso) {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.round(minutes / 60);
  return `${hours} hour${hours === 1 ? "" : "s"} ago`;
}

function editionLine() {
  const live = state.meta.live;
  if (live && live.last_refresh) return `Live edition, feeds checked ${minutesAgo(new Date(live.last_refresh * 1000).toISOString())}`;
  return `Edition of ${editionDate(state.meta.as_of)}`;
}

function renderEdition(reader) {
  $("#edition").replaceChildren(`${editionLine()}, ranked for `, el("strong", { text: reader.name }));
}

/* Live mode: the server rebuilds its index when feeds bring new stories.
   Offer them instead of reshuffling the page under the reader. */
async function pollEdition() {
  if (document.visibilityState === "hidden") return;
  try {
    const meta = await api("/api/meta");
    const fresh = meta.edition !== state.meta.edition;
    state.meta = meta;
    if (state.feed) renderEdition(state.feed.reader);
    if (fresh && state.readerId) toast("New stories are in.", "Show them", () => loadFeed({ animate: true }), 15000);
  } catch { /* try again next time */ }
}

function storyNode(item, position) {
  const node = $("#story-template").content.firstElementChild.cloneNode(true);
  node.dataset.id = item.id;
  if (position === 0) node.classList.add("lead");
  if (item.discovery) node.classList.add("is-discovery");
  $(".rank", node).textContent = item.rank;

  const category = item.categories[0]?.name || "General";
  const kicker = $(".kicker", node);
  if (item.discovery) kicker.append(el("span", { class: "discovery", text: "Discovery" }), ", ");
  kicker.append(category);

  const link = $(".headline a", node);
  link.href = item.url || "#";
  link.textContent = item.title;
  if (isHindi(item.title)) node.setAttribute("lang", "hi");
  link.addEventListener("click", () => send(item.id, "open", { quiet: true }));

  $(".summary", node).textContent = item.summary;
  const meta = $(".meta", node);
  meta.append(el("b", { text: item.source || "Unknown outlet" }), `, ${age(item.age_hours)}`);
  if (item.coverage > 1) meta.append(`, reported by ${item.coverage} outlets`);

  $(".reasons", node).append(...item.reasons.map((r) => el("li", { text: r })));

  if (position > 0 && item.image) {
    // Publisher thumbnails, toned to the palette; a broken image just disappears.
    const thumb = el("img", { class: "thumb", src: item.image, alt: "", loading: "lazy", referrerpolicy: "no-referrer",
      onerror: (e) => { e.target.remove(); node.classList.remove("has-thumb"); } });
    node.classList.add("has-thumb");
    $(".story-body", node).prepend(thumb);
  }
  if (position === 0 && item.image) {
    const figure = el("figure", { class: "lead-figure" },
      el("img", { src: item.image, alt: "", loading: "lazy", referrerpolicy: "no-referrer", onerror: (e) => e.target.closest("figure").remove() }));
    $(".meta", node).after(figure);
  }

  const like = $('[data-action="like"]', node);
  const save = $('[data-action="save"]', node);
  like.setAttribute("aria-pressed", String(item.liked));
  save.setAttribute("aria-pressed", String(item.saved));
  like.setAttribute("aria-label", `Like: ${item.title}`);
  save.setAttribute("aria-label", `Save: ${item.title}`);
  $('[data-action="hide"]', node).setAttribute("aria-label", `Not for me: ${item.title}`);
  like.addEventListener("click", () => toggle(item, "like", like));
  save.addEventListener("click", () => toggle(item, "save", save));
  $('[data-action="hide"]', node).addEventListener("click", () => hide(item, node));

  const why = $(".why", node);
  const panel = $(".why-panel", node);
  why.addEventListener("click", () => {
    const open = why.getAttribute("aria-expanded") === "true";
    why.setAttribute("aria-expanded", String(!open));
    if (!open && !panel.childElementCount) panel.append(...whyContent(item));
    panel.hidden = open;
  });
  return node;
}

function whyContent(item) {
  const weights = state.feed.weights;
  const max = Math.max(0.01, ...Object.values(item.contributions).map(Math.abs));
  const rows = Object.keys(SIGNAL_NAMES).map((signal) => {
    const active = signal in item.contributions;
    const value = active ? item.contributions[signal] : 0;
    return el("li", {},
      el("span", { class: active ? null : "off", text: SIGNAL_NAMES[signal] }),
      el("span", { class: "track", title: `signal ${item.signals[signal]}, weight ${weights[signal]}` },
        active ? el("i", { class: value < 0 ? "neg" : null, style: `width:${(Math.abs(value) / max) * 100}%` }) : null),
      el("span", { class: "num", text: active ? (value * 100).toFixed(1) : "off" }),
    );
  });
  const out = [
    el("h4", { text: `Match ${item.match} of 100, built from` }),
    el("ul", { class: "contrib" }, rows),
  ];
  const moved = item.base_rank - item.rank;
  if (item.discovery) {
    out.push(el("p", { text: `Placed in a discovery slot. On score alone it would sit at ${item.base_rank}.` }));
  } else if (moved > 0) {
    out.push(el("p", { text: `Moved up ${moved} place${moved === 1 ? "" : "s"} from ${item.base_rank} because the stories it overtook repeated a topic or outlet already above.` }));
  } else if (moved < 0) {
    out.push(el("p", { text: `Moved down ${-moved} from ${item.base_rank} to make room for more variety.` }));
  } else {
    out.push(el("p", { class: "muted", text: "Same place as on score alone." }));
  }
  const inactive = Object.keys(SIGNAL_NAMES).filter((s) => !(s in item.contributions));
  if (inactive.length) {
    out.push(el("p", { class: "muted", text: `Not used yet for this reader: ${inactive.map((s) => SIGNAL_NAMES[s].toLowerCase()).join(", ")}.` }));
  }
  if (item.related.length) {
    out.push(
      el("p", { text: "The same story from other outlets:" }),
      el("ul", { class: "coverage" }, item.related.map((r) =>
        el("li", {}, el("a", { href: r.url, target: "_blank", rel: "noopener noreferrer", text: r.title }), ` (${r.source || "unknown outlet"}${r.sentiment ? `, ${r.sentiment} tone` : ""})`))),
    );
  }
  return out;
}

function renderFeed(feed, animate) {
  const list = $("#feed-list");
  const before = new Map();
  if (animate) for (const node of list.children) if (node.dataset.id) before.set(node.dataset.id, node.getBoundingClientRect().top);

  if (!feed.items.length) {
    const filtered = state.category ? " in this category" : "";
    list.replaceChildren(el("li", { class: "empty" },
      el("h3", { text: `Nothing left to show${filtered}` }),
      el("p", { text: state.category ? "Choose All stories, or turn off Hide stories I already opened." : "Turn off Hide stories I already opened, or clear the reading history." })));
    $("#feed-foot").textContent = "";
    return;
  }
  const nodes = feed.items.map(storyNode);
  list.replaceChildren(...nodes);

  if (animate && before.size) {
    for (const node of nodes) {
      const old = before.get(node.dataset.id);
      if (old === undefined) {
        node.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 260, easing: "ease-out" });
        continue;
      }
      const delta = old - node.getBoundingClientRect().top;
      if (Math.abs(delta) > 2) {
        node.animate([{ transform: `translateY(${delta}px)` }, { transform: "none" }], { duration: 380, easing: "cubic-bezier(.2,.8,.2,1)" });
      }
    }
  }
  const discovery = feed.items.filter((i) => i.discovery).length;
  $("#feed-foot").textContent =
    `${feed.items.length} of ${feed.eligible.toLocaleString("en-IN")} stories that fit this reader` +
    (discovery ? `, ${discovery} of them discovery picks` : "") + `. Ranked in ${feed.elapsed_ms} ms.`;
  observeViews(nodes);
}

/* ------------------------------------------------------------------ */
/* Shape panel                                                         */
/* ------------------------------------------------------------------ */

function renderShape(feed) {
  const shape = feed.shape;
  const holder = $("#shape");
  if (!feed.items.length) { holder.replaceChildren(el("p", { class: "shape-note", text: "No stories in the feed." })); return; }
  const total = feed.items.length;
  const top = shape.categories.slice(0, 7);
  const rest = shape.categories.slice(7).reduce((n, c) => n + c.count, 0);
  if (rest) top.push({ key: "_other", name: "Other", count: rest });
  const mix = el("div", { class: "mix", role: "img", "aria-label": top.map((c) => `${c.name} ${c.count}`).join(", ") },
    top.map((c, i) => el("span", { style: `flex:${c.count};background:${MIX_COLOURS[i % MIX_COLOURS.length]}`, title: `${c.name}: ${c.count}` })));
  const legend = el("ul", { class: "mix-legend" },
    top.map((c, i) => el("li", {}, el("i", { style: `background:${MIX_COLOURS[i % MIX_COLOURS.length]}` }), c.name, el("span", { class: "n", text: c.count }))));

  const tones = ["negative", "neutral", "positive", "unknown"].filter((t) => shape.sentiment[t]);
  const figures = el("dl", { class: "figures" },
    el("div", {}, el("dt", { text: "Categories" }), el("dd", { text: shape.categories.length })),
    el("div", {}, el("dt", { text: "Outlets" }), el("dd", { text: shape.sources })),
    el("div", {}, el("dt", { text: "Median age" }), el("dd", {}, String(Math.round(shape.median_age_hours)), el("small", { text: "hours" }))),
    el("div", {}, el("dt", { text: "Overlap" }), el("dd", {}, String(Math.round(shape.similarity * 100)), el("small", { text: "%" }))),
  );
  holder.replaceChildren(
    mix, legend, figures,
    el("p", { class: "tone-legend", text: "Tone of the stories" }),
    el("div", { class: "tone", role: "img", "aria-label": tones.map((t) => `${shape.sentiment[t]} ${t}`).join(", ") },
      tones.map((t) => el("span", { class: `tone-${t}`, style: `flex:${shape.sentiment[t]}`, title: `${t}: ${shape.sentiment[t]}` }))),
    el("p", { class: "tone-legend", text: tones.map((t) => `${shape.sentiment[t]} ${t}`).join(", ") }),
    el("p", { class: "shape-note", text: `Overlap is the average text similarity between any two stories in the feed. ${total} stories shown.` }),
  );
}

function renderMethod() {
  const steps = [
    ["Six signals", " score every story: category, topic tags, similarity to what you opened, freshness, outlet and coverage."],
    ["Weights blend them.", " A signal with nothing to say yet, such as similarity before your first click, sits out."],
    ["Repeats are spaced out.", " A story loses ground if the slots above it already cover its topic or outlet."],
    ["Discovery slots", " bring in well-covered stories from categories you have not read, so the feed does not only echo you."],
    ["Your actions", " (open, like, save, not for me) move the next feed. Impressions you scroll past slowly sink."],
  ];
  $("#method").replaceChildren(...steps.map(([b, rest]) => el("li", {}, el("b", { text: b }), rest)));
}

/* ------------------------------------------------------------------ */
/* Interactions                                                        */
/* ------------------------------------------------------------------ */

async function send(articleId, action, { quiet = false, refresh = true } = {}) {
  try {
    await api(`/api/readers/${state.readerId}/events`, { method: "POST", body: { events: [{ article_id: articleId, action }] } });
    if (refresh) setTimeout(() => loadFeed({ animate: true }), quiet ? 600 : 0);
  } catch (error) {
    toast(`That did not save: ${error.message}`);
  }
}

async function toggle(item, action, button) {
  const on = button.getAttribute("aria-pressed") !== "true";
  button.setAttribute("aria-pressed", String(on));
  try {
    if (on) await api(`/api/readers/${state.readerId}/events`, { method: "POST", body: { events: [{ article_id: item.id, action }] } });
    else await api(`/api/readers/${state.readerId}/events`, { method: "DELETE", body: { article_id: item.id, action } });
    toast(on ? (action === "like" ? "Liked. Similar stories will rank higher." : "Saved") : (action === "like" ? "Like removed" : "Removed from saved"));
    loadFeed({ animate: true });
  } catch (error) {
    button.setAttribute("aria-pressed", String(!on));
    toast(`That did not save: ${error.message}`);
  }
}

async function hide(item, node) {
  node.classList.add("leaving");
  try {
    await api(`/api/readers/${state.readerId}/events`, { method: "POST", body: { events: [{ article_id: item.id, action: "hide" }] } });
  } catch (error) {
    node.classList.remove("leaving");
    toast(`That did not save: ${error.message}`);
    return;
  }
  toast("Hidden. You will see less like it.", "Undo", async () => {
    await api(`/api/readers/${state.readerId}/events`, { method: "DELETE", body: { article_id: item.id, action: "hide" } });
    loadFeed({ animate: true });
  });
  setTimeout(() => loadFeed({ animate: true }), reduceMotion ? 0 : 240);
}

/* Impressions: a story counts as seen after it is mostly on screen for a second. */
let viewObserver = null;
function observeViews(nodes) {
  if (!("IntersectionObserver" in window)) return;
  viewObserver?.disconnect();
  const timers = new Map();
  viewObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const id = entry.target.dataset.id;
      if (entry.isIntersecting) {
        if (state.seen.has(id) || timers.has(id)) continue;
        timers.set(id, setTimeout(() => {
          state.seen.add(id);
          state.pendingViews.push(id);
          timers.delete(id);
        }, 1000));
      } else if (timers.has(id)) {
        clearTimeout(timers.get(id));
        timers.delete(id);
      }
    }
  }, { threshold: 0.6 });
  nodes.forEach((n) => viewObserver.observe(n));
}

function flushViews() {
  if (!state.pendingViews.length || !state.readerId) return;
  const events = state.pendingViews.splice(0).map((id) => ({ article_id: id, action: "view" }));
  const body = JSON.stringify({ events });
  const url = `/api/readers/${state.readerId}/events`;
  if (document.visibilityState === "hidden" && navigator.sendBeacon) {
    navigator.sendBeacon(url, new Blob([body], { type: "application/json" }));
  } else {
    fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: true }).catch(() => {});
  }
}

let toastTimer = null;
function toast(message, actionLabel, action, duration) {
  const box = $("#toast");
  box.replaceChildren(el("span", { text: message }));
  if (actionLabel) {
    box.append(el("button", { type: "button", text: actionLabel, onclick: () => { box.hidden = true; action(); } }));
  }
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { box.hidden = true; }, duration || (actionLabel ? 6000 : 2800));
}

/* ------------------------------------------------------------------ */
/* Phone sheets                                                        */
/* ------------------------------------------------------------------ */

function setupSheets() {
  const scrim = $("#scrim");
  const buttons = [...document.querySelectorAll(".dock button")];
  const close = () => {
    document.querySelectorAll(".panel.open").forEach((p) => p.classList.remove("open"));
    buttons.forEach((b) => b.setAttribute("aria-expanded", "false"));
    scrim.hidden = true;
  };
  for (const button of buttons) {
    button.setAttribute("aria-controls", button.dataset.sheet);
    button.addEventListener("click", () => {
      const panel = document.getElementById(button.dataset.sheet);
      const wasOpen = panel.classList.contains("open");
      close();
      if (wasOpen) return;
      panel.classList.add("open");
      button.setAttribute("aria-expanded", "true");
      scrim.hidden = false;
      panel.scrollTop = 0;
    });
  }
  scrim.addEventListener("click", close);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
  window.matchMedia("(min-width: 761px)").addEventListener("change", close);
}

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */

async function boot() {
  drawFan();
  setupSheets();
  try {
    state.meta = await api("/api/meta");
    await refreshReaders();
  } catch (error) {
    $("#edition").textContent = "The server did not answer.";
    $("#feed-list").replaceChildren(el("li", { class: "error-box" }, `Could not reach the Fanlight API (${error.message}). Start it with: fanlight serve`));
    return;
  }
  const prefs = loadPrefs();
  state.tuning = prefs.tuning || {};
  const known = state.readers.some((r) => r.id === prefs.readerId);
  state.readerId = known ? prefs.readerId : (state.readers.find((r) => r.interest_count > 0) || state.readers[0])?.id;

  buildDials();
  renderFilters();
  renderMethod();
  setupDialog();
  $("#reader-select").addEventListener("change", (e) => selectReader(e.target.value));
  renderReaderSelect();
  setInterval(flushViews, 4000);
  document.addEventListener("visibilitychange", flushViews);
  if (state.meta.live) setInterval(pollEdition, META_POLL_MS);
  if (!state.readerId) {
    showWelcome();
    openReaderDialog(true);
    return;
  }
  await loadFeed({ animate: false });
}

boot();
