
-- This just collapses all callouts by default, could modify if we wanted to collapse non-solution callouts differently

function Callout(el)
  if quarto.doc.isFormat("html") then
    -- Set default collapse to true if unset
    if "sol" == el.type then
      if not el.collapse then
        el.collapse = true
      end
    end
    return el
  end
end