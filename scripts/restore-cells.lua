--- @module "restore-cells"
--- @license MIT
--- @copyright 2026 Mickaël Canouil
--- @author Mickaël Canouil
--- @version 1.0.0
--- @brief Restore cell structure for CodeBlocks inside cross-reference divs
--- @description Extracts cell Divs from nested wrapper Divs and wraps bare CodeBlocks
--- so Pandoc's ipynb writer sees them correctly as code cells.

-- Only run for ipynb output
if not quarto.doc.is_format("ipynb") then
  return {}
end

--- Check if a CodeBlock should be wrapped in a cell
--- @param block table
--- @return boolean
local function is_unwrapped_cell_code(block)
  if block.t ~= "CodeBlock" then
    return false
  end
  return block.classes:includes("cell-code")
end

--- Wrap a CodeBlock in a cell Div
--- @param codeblock pandoc.CodeBlock
--- @return pandoc.Div
local function wrap_in_cell(codeblock)
  return pandoc.Div(
    pandoc.Blocks({ codeblock }),
    pandoc.Attr("", { "cell" }, {})
  )
end

--- Check if a Div is a cell
--- @param block table
--- @return boolean
local function is_cell_div(block)
  return block.t == "Div" and block.classes:includes("cell")
end

--- Process blocks recursively
--- @param blocks pandoc.Blocks
--- @return pandoc.Blocks
local function process_blocks(blocks)
  local result = pandoc.Blocks({})

  for _, block in ipairs(blocks) do
    if is_unwrapped_cell_code(block) then
      -- Bare CodeBlock with cell-code class - wrap it
      result:insert(wrap_in_cell(block))
    elseif block.t == "Div" and not is_cell_div(block) and block.content then
      -- Non-cell Div - process content recursively and check for nested cells to extract
      local processed = process_blocks(block.content)

      -- Separate cells from other content
      local cells = pandoc.Blocks({})
      local other = pandoc.Blocks({})

      for _, inner in ipairs(processed) do
        if is_cell_div(inner) then
          cells:insert(inner)
        else
          other:insert(inner)
        end
      end

      -- If we found cells, extract them
      if #cells > 0 then
        if #other > 0 then
          block.content = other
          result:insert(block)
        end
        for _, cell in ipairs(cells) do
          result:insert(cell)
        end
      else
        block.content = processed
        result:insert(block)
      end
    else
      result:insert(block)
    end
  end

  return result
end

-- Main filter
return {
  {
    Pandoc = function(doc)
      doc.blocks = process_blocks(doc.blocks)
      return doc
    end
  }
}