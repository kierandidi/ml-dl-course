# Fix kramdown-math-katex FrozenError when Katex.render returns frozen strings
Jekyll::Hooks.register :site, :after_init do |_site|
  require 'kramdown/converter/math_engine/katex'

  module Kramdown::Converter::MathEngine::Katex
    def self.call(converter, el, opts)
      display_mode = el.options[:category] == :block
      result = ::Katex.render(
        el.value,
        display_mode: display_mode,
        throw_on_error: false,
        **converter.options[:math_engine_opts]
      )
      result = result.dup
      attr = el.attr.dup
      attr.delete('xmlns')
      attr.delete('display')
      result.insert(result =~ /[[:space:]>]/, converter.html_attributes(attr))
      result = "#{' ' * opts[:indent]}#{result}\n" if display_mode
      result
    end
  end
end
