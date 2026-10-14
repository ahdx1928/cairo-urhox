-- Surrounding terrain: desert, mountains and sea.
--
-- The heightfield and the triangle mesh are both produced offline; this module
-- only loads the mesh and picks a material per chunk. Reusing the city mesh
-- format means the same loader, vertex layout and buffer path apply.
--
-- Chunk styles, matching city-bake/bake_terrain.py:
--   300 sand   301 rock   302 water

local CityMesh = require("city.CityMesh")

local Terrain = {}

local STYLE_SAND = 300
local STYLE_ROCK = 301
local STYLE_WATER = 302

---@param parent Node
---@param chunks MeshChunk[]     from CityMesh.Load
---@param materials table<integer, Material>  keyed by style
---@return integer count
function Terrain.Attach(parent, chunks, materials)
    local root = parent:CreateChild("Terrain")
    local n = 0
    for i = 1, #chunks do
        local chunk = chunks[i]
        local mat = materials[chunk.style] or materials[STYLE_SAND]
        local node = root:CreateChild("T" .. i)
        local sm = node:CreateComponent("StaticModel")
        sm:SetModel(CityMesh.MakeModel(chunk))
        sm:SetMaterial(mat)
        -- distant geometry: shadow casting would cost a pass for no gain
        sm.castShadows = false
        n = n + 1
    end
    return n
end

---Load the terrain mesh and attach it.
---@param parent Node
---@param loadMesh fun(name: string): MeshChunk[]
---@param materials table<integer, Material>
---@return integer count
function Terrain.Build(parent, loadMesh, materials)
    local chunks = loadMesh("cairo_terrain_mesh.bin")
    if #chunks == 0 then
        log:Write(LOG_WARNING, "[Terrain] no terrain chunks loaded")
        return 0
    end
    local n = Terrain.Attach(parent, chunks, materials)
    log:Write(LOG_INFO, string.format("[Terrain] %d chunks attached", n))
    return n
end

Terrain.STYLE_SAND = STYLE_SAND
Terrain.STYLE_ROCK = STYLE_ROCK
Terrain.STYLE_WATER = STYLE_WATER

return Terrain
