# Render KaTeX at build time on an ES5 JS engine (duktape).
#
# Two problems are fixed here:
#   1. duktape is ES5.1 and lacks several ES6 builtins. KaTeX's matrix / array
#      environments (pmatrix, aligned, cases, bmatrix, ...) call
#      `Array.prototype.fill`, which throws
#      "undefined not callable (property 'fill' of [object Array])".
#      We inject small ES5 polyfills before KaTeX loads.
#   2. kramdown-math-katex returns frozen strings (FrozenError on insert).
#      We build our own ExecJS context and render directly.
require "execjs"

module CourseKatex
  # ES5-safe polyfills for the builtins KaTeX relies on but duktape lacks.
  POLYFILL = <<~'JS'
    (function () {
      if (!Array.prototype.fill) {
        Object.defineProperty(Array.prototype, 'fill', {
          configurable: true, writable: true,
          value: function (value) {
            var O = Object(this), len = O.length >>> 0;
            var start = arguments[1] >> 0;
            var k = start < 0 ? Math.max(len + start, 0) : Math.min(start, len);
            var end = arguments[2] === undefined ? len : arguments[2] >> 0;
            var fin = end < 0 ? Math.max(len + end, 0) : Math.min(end, len);
            while (k < fin) { O[k] = value; k++; }
            return O;
          }
        });
      }
      if (!Array.from) {
        Array.from = function (arrayLike, mapFn) {
          var out = [], len = arrayLike.length >>> 0;
          for (var i = 0; i < len; i++) out.push(mapFn ? mapFn(arrayLike[i], i) : arrayLike[i]);
          return out;
        };
      }
      if (!Array.prototype.includes) {
        Array.prototype.includes = function (s) { return this.indexOf(s) !== -1; };
      }
      if (!Array.prototype.find) {
        Array.prototype.find = function (pred) {
          for (var i = 0; i < this.length; i++) if (pred(this[i], i, this)) return this[i];
          return undefined;
        };
      }
      if (!Array.prototype.findIndex) {
        Array.prototype.findIndex = function (pred) {
          for (var i = 0; i < this.length; i++) if (pred(this[i], i, this)) return i;
          return -1;
        };
      }
      if (!String.prototype.repeat) {
        String.prototype.repeat = function (n) {
          var s = '', x = '' + this; n = n >> 0;
          while (n > 0) { if (n & 1) s += x; n >>= 1; if (n) x += x; }
          return s;
        };
      }
      if (!String.prototype.startsWith) {
        String.prototype.startsWith = function (p, pos) { return this.indexOf(p, pos || 0) === (pos || 0); };
      }
      if (!String.prototype.endsWith) {
        String.prototype.endsWith = function (p, len) {
          if (len === undefined || len > this.length) len = this.length;
          return this.substring(len - p.length, len) === p;
        };
      }
      if (!Object.assign) {
        Object.assign = function (target) {
          for (var i = 1; i < arguments.length; i++) {
            var src = arguments[i];
            if (src) for (var k in src) if (Object.prototype.hasOwnProperty.call(src, k)) target[k] = src[k];
          }
          return target;
        };
      }
      if (!Number.isNaN) { Number.isNaN = function (v) { return typeof v === 'number' && v !== v; }; }
      if (!Number.isFinite) { Number.isFinite = function (v) { return typeof v === 'number' && isFinite(v); }; }
    })();
  JS

  def self.katex_js_path
    @katex_js_path ||= begin
      spec = Gem::Specification.find_by_name("katex")
      candidates =
        Dir.glob(File.join(spec.gem_dir, "**", "katex.min.js")) +
        Dir.glob(File.join(spec.gem_dir, "**", "katex.js"))
      candidates.first
    end
  end

  def self.context
    @context ||= ExecJS.compile(POLYFILL + "\n" + File.read(katex_js_path))
  end

  def self.render(tex, display_mode)
    context.call(
      "katex.renderToString",
      tex.to_s,
      "displayMode" => display_mode, "throwOnError" => false,
    )
  rescue => e
    "<span class=\"katex-error\" title=\"#{e.message}\">#{tex}</span>"
  end
end

Jekyll::Hooks.register :site, :after_init do |_site|
  require "kramdown/converter/math_engine/katex"

  module Kramdown::Converter::MathEngine::Katex
    def self.call(converter, el, opts)
      display_mode = el.options[:category] == :block
      result = CourseKatex.render(el.value, display_mode).dup
      attr = el.attr.dup
      attr.delete("xmlns")
      attr.delete("display")
      pos = (result =~ /[[:space:]>]/)
      result.insert(pos, converter.html_attributes(attr)) if pos
      result = "#{' ' * opts[:indent]}#{result}\n" if display_mode
      result
    end
  end
end
