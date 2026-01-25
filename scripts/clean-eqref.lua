return {
    Span = function(el)

        local found = false
        for i, block in ipairs(el.content) do
            if block.t == "Math" then
                return block
            end
        end 

        return el


    end
}
