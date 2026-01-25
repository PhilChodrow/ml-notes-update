return {
    {
        CodeBlock = function(el)
            
            local lines = pandoc.List()
            local code = el.text

            local trimmed = string.gsub(code, "#%-%-%-.-#%-%-%-", "# TODO")  
            
            el.text = trimmed

            return el
            
        end
    }
}