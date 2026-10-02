// 外语课中文文稿的朗读条：浏览器自带 Web Speech API 现场合成，不预先生成音频。
// build_book.sh 只给源语言不是中文的课插 <div class="v2t-tts">，没有标记的页面什么都不做。
(function () {
  "use strict";

  // 按句切块：Chrome 的在线语音一条念太长会卡死，句子级也方便暂停后从本句重来
  function split(text) {
    var out = [];
    (text.match(/[^。！？；!?;]+[。！？；!?;”"）)]*/g) || []).forEach(function (s) {
      s = s.trim();
      while (s.length > 160) {
        var cut = Math.max(s.lastIndexOf("，", 160), s.lastIndexOf(",", 160));
        cut = cut > 40 ? cut + 1 : 160;
        out.push(s.slice(0, cut));
        s = s.slice(cut).trim();
      }
      if (s) out.push(s);
    });
    return out;
  }

  // 只念正文：去掉回视频的时间链接（或本地源的纯时间 *12:34*）、配图、代码块、嵌套列表
  var STAMP = /^\s*\d{1,2}:\d{2}(:\d{2})?\s*$/;
  function textOf(el) {
    var c = el.cloneNode(true);
    c.querySelectorAll("img, pre, ul, ol, sup").forEach(function (n) { n.remove(); });
    c.querySelectorAll("a, em").forEach(function (n) {
      if (STAMP.test(n.textContent)) n.remove();
    });
    return c.textContent.replace(/\s+/g, " ").trim();
  }

  function units(root) {
    return Array.prototype.filter.call(
      root.querySelectorAll("h1, h2, h3, h4, p, li"),
      function (el) {
        if (el.tagName === "P" && el.closest("li")) return false; // 交给外层 li
        return textOf(el).length > 0;
      }
    );
  }

  if (typeof module !== "undefined") module.exports = { split: split, textOf: textOf, units: units };
  if (typeof document === "undefined") return;

  function mount() {
    var marks = document.querySelectorAll(".v2t-tts");
    // print.html 把所有章拼在一页，朗读条没意义
    if (/print\.html$/.test(location.pathname)) {
      marks.forEach(function (m) { m.remove(); });
      return;
    }
    var bar = marks[0];
    if (!bar) return;
    if (!("speechSynthesis" in window) || typeof SpeechSynthesisUtterance === "undefined") {
      bar.textContent = "这个浏览器不支持朗读（Web Speech API），换 Edge / Chrome / Safari 试试。";
      return;
    }
    var synth = window.speechSynthesis;
    var root = document.querySelector("main") || document.body;
    var list = units(root);
    if (!list.length) { bar.remove(); return; }

    bar.innerHTML =
      '<button type="button" class="v2t-play" aria-label="朗读中文文稿">▶ 朗读</button>' +
      '<button type="button" class="v2t-stop" aria-label="停止朗读" disabled>■</button>' +
      '<label>语速 <select class="v2t-rate">' +
      [0.8, 1, 1.2, 1.5, 1.8, 2].map(function (r) {
        return '<option value="' + r + '">' + r + "x</option>";
      }).join("") +
      "</select></label>" +
      '<label>声音 <select class="v2t-voice"></select></label>' +
      '<span class="v2t-status" role="status" aria-live="polite"></span>';
    var $ = function (s) { return bar.querySelector(s); };
    var playBtn = $(".v2t-play"), stopBtn = $(".v2t-stop");
    var rateSel = $(".v2t-rate"), voiceSel = $(".v2t-voice"), status = $(".v2t-status");

    var store = {
      get: function (k) { try { return localStorage.getItem("v2t-tts-" + k); } catch (e) { return null; } },
      set: function (k, v) { try { localStorage.setItem("v2t-tts-" + k, v); } catch (e) { /* 隐私模式 */ } },
    };
    rateSel.value = store.get("rate") || "1.2";
    if (!rateSel.value) rateSel.value = "1.2"; // 存的值不在选项里时 value 会变成空

    var voices = [];
    function loadVoices() {
      voices = synth.getVoices().filter(function (v) { return /^(zh|cmn)/i.test(v.lang); });
      // 优先普通话，其次在线/神经网络音色（Edge 的 Xiaoxiao 等明显更自然）
      var score = function (v) {
        return (/zh[-_]CN|cmn/i.test(v.lang) ? 4 : 0) +
          (/natural|neural|online|google/i.test(v.name) ? 2 : 0) + (v.localService ? 0 : 1);
      };
      voices.sort(function (a, b) { return score(b) - score(a); });
      var saved = store.get("voice");
      voiceSel.replaceChildren();
      voices.forEach(function (v, i) { voiceSel.add(new Option(v.name, String(i), false, v.name === saved)); });
      if (!voices.length) voiceSel.add(new Option("系统默认", ""));
      status.textContent = voices.length ? "" : "没找到中文语音，可能念不出来；Edge / Chrome 自带，系统里也可以装中文语音包。";
    }
    loadVoices();
    if ("onvoiceschanged" in synth) synth.addEventListener("voiceschanged", loadVoices);

    var idx = -1, chunks = [], ci = 0, playing = false, gen = 0;

    function highlight(on) {
      list.forEach(function (el) { el.classList.remove("v2t-reading"); });
      if (on && list[idx]) {
        list[idx].classList.add("v2t-reading");
        var r = list[idx].getBoundingClientRect();
        if (r.top < 0 || r.bottom > window.innerHeight) {
          list[idx].scrollIntoView({ block: "center", behavior: "smooth" });
        }
      }
    }

    function load(i) {
      idx = i; ci = 0;
      chunks = split(textOf(list[i]));
      highlight(true);
      status.textContent = "第 " + (i + 1) + " / " + list.length + " 段";
    }

    function speak() {
      var my = gen;
      while (idx < list.length && ci >= chunks.length) {
        if (idx + 1 >= list.length) { stop(); status.textContent = "念完了"; return; }
        load(idx + 1);
      }
      var u = new SpeechSynthesisUtterance(chunks[ci]);
      u.lang = "zh-CN";
      u.rate = parseFloat(rateSel.value) || 1;
      var v = voices[+voiceSel.value];
      if (v) u.voice = v;
      u.onend = function () {
        if (my !== gen || !playing) return; // 被暂停/停止 cancel 掉的那句
        ci++; speak();
      };
      u.onerror = function (e) {
        if (my !== gen || e.error === "interrupted" || e.error === "canceled") return;
        stop();
        status.textContent = "朗读出错（" + e.error + "），换个声音或浏览器试试。";
      };
      synth.speak(u);
    }

    // 没开始过就从屏幕上第一段可见的地方念起
    function firstVisible() {
      for (var i = 0; i < list.length; i++) {
        if (list[i].getBoundingClientRect().bottom > 0) return i;
      }
      return 0;
    }

    function play() {
      playing = true; gen++;
      if (idx < 0) load(firstVisible());
      synth.cancel();
      playBtn.textContent = "❚❚ 暂停";
      playBtn.setAttribute("aria-label", "暂停朗读");
      stopBtn.disabled = false;
      // Chrome：cancel() 之后同一拍里的 speak() 偶尔被吞
      setTimeout(speak, 50);
    }

    function pause() {
      // 不用 synth.pause()：Chrome 的在线语音暂停后常常恢复不了，改成停掉、恢复时重念本句
      playing = false; gen++;
      synth.cancel();
      playBtn.textContent = "▶ 继续";
      playBtn.setAttribute("aria-label", "继续朗读");
    }

    function stop() {
      pause();
      idx = -1;
      highlight(false);
      playBtn.textContent = "▶ 朗读";
      playBtn.setAttribute("aria-label", "朗读中文文稿");
      stopBtn.disabled = true;
      status.textContent = "";
    }

    playBtn.addEventListener("click", function () { playing ? pause() : play(); });
    stopBtn.addEventListener("click", stop);
    // 语速/声音从下一句开始生效
    rateSel.addEventListener("change", function () { store.set("rate", rateSel.value); });
    voiceSel.addEventListener("change", function () {
      var v = voices[+voiceSel.value];
      if (v) store.set("voice", v.name);
    });
    // 双击某段从那段开始念
    list.forEach(function (el, i) {
      el.addEventListener("dblclick", function () { load(i); play(); });
    });
    window.addEventListener("pagehide", function () { synth.cancel(); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
