-- after https://github.com/insightsengineering/pattern-strip/blob/main/_extensions/pattern-strip/pattern-strip.lua

return {
    {
        CodeBlock = function(el)
                        
            local lines = pandoc.List()
            local code = el.text .. "\n"

            for line in code:gmatch("([^\n]*)\n") do
                if string.find(line, "%-%-%-") == nil then
                    lines:insert(line)
                end
            end
            
            el.text = table.concat(lines, "\n")

            return el
            
        end
    }
}