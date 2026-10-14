-- Loads the pre-baked Cairo mesh and hands back engine buffers.
--
-- The heavy lifting (extrusion, triangulation, sun shading) happens offline in
-- Python, so this module never touches individual vertices. Each chunk is one
-- StaticModel sharing a single unlit-vertex-colour material.
--
-- Vertex layout baked into the file: position (3 x f32) then colour (4 x u8).
-- The sun term is already folded into the colour.

local CityMesh = {}

local MAGIC = "URBXMSH1"
local VERTEX_SIZE = 16

---@class MeshChunk
---@field vb VertexBuffer
---@field ib IndexBuffer
---@field nv integer
---@field ni integer
---@field layout integer   0 = position+colour, 1 = position+uv, 2 = position+normal+uv
---@field style integer    facade variant index (buildings only)
---@field min Vector3
---@field max Vector3

---Opens the mesh binary through the resource cache. A raw `File` cannot see
---packaged assets: the virtual filesystem returns a zero-length handle for a
---path that has not been materialised yet, while `cache:GetFile` resolves it to
---the real packaged file.
---@param name string
---@return File|nil
local function openAsset(name)
    local path = "City/" .. name
    local f = cache:GetFile(path)
    if f then
        return f
    end
    log:Write(LOG_ERROR, "[CityMesh] cannot open " .. path)
    return nil
end

---@param f File
---@param expected string
---@return boolean
local function checkMagic(f, expected)
    if f:GetSize() < #expected then
        return false
    end
    for i = 1, #expected do
        if f:ReadUByte() ~= string.byte(expected, i) then
            return false
        end
    end
    return true
end

---@param name string
---@return MeshChunk[]
function CityMesh.Load(name)
    local chunks = {}
    local f = openAsset(name)
    if not f then
        return chunks
    end
    if not checkMagic(f, MAGIC) then
        log:Write(LOG_ERROR, "[CityMesh] bad magic in " .. name)
        f:Close()
        return chunks
    end

    -- Three vertex layouts; the file flags which one each chunk uses so the
    -- right element set is declared here.
    local elementsByLayout = {
        [0] = {
            VertexElement(TYPE_VECTOR3, SEM_POSITION),
            VertexElement(TYPE_UBYTE4_NORM, SEM_COLOR),
        },
        [1] = {
            VertexElement(TYPE_VECTOR3, SEM_POSITION),
            VertexElement(TYPE_VECTOR2, SEM_TEXCOORD),
        },
        [2] = {
            VertexElement(TYPE_VECTOR3, SEM_POSITION),
            VertexElement(TYPE_VECTOR3, SEM_NORMAL),
            VertexElement(TYPE_VECTOR2, SEM_TEXCOORD),
        },
    }

    local t0 = os.clock()
    local nChunks = f:ReadUInt()
    local totalVerts = 0
    local totalTris = 0
    for i = 1, nChunks do
        local vbytes = f:ReadUInt()
        local ibytes = f:ReadUInt()
        local nv = f:ReadUInt()
        local ni = f:ReadUInt()
        local layout = f:ReadUInt()
        local style = f:ReadUInt()
        f:ReadUInt()  -- reserved
        local minX, minY, minZ = f:ReadFloat(), f:ReadFloat(), f:ReadFloat()
        local maxX, maxY, maxZ = f:ReadFloat(), f:ReadFloat(), f:ReadFloat()

        local elements = elementsByLayout[layout]
        if not elements then
            log:Write(LOG_ERROR, "[CityMesh] unknown vertex layout " .. layout .. " on chunk " .. i)
            break
        end

        local vb = VertexBuffer()
        if not vb:SetSize(nv, elements, false) then
            log:Write(LOG_ERROR, "[CityMesh] VertexBuffer:SetSize failed on chunk " .. i)
            break
        end
        if not vb:SetData(f:Read(vbytes)) then
            log:Write(LOG_ERROR, "[CityMesh] VertexBuffer:SetData failed on chunk " .. i)
            break
        end

        local ib = IndexBuffer()
        if not ib:SetSize(ni, false, false) then
            log:Write(LOG_ERROR, "[CityMesh] IndexBuffer:SetSize failed on chunk " .. i)
            break
        end
        if not ib:SetData(f:Read(ibytes)) then
            log:Write(LOG_ERROR, "[CityMesh] IndexBuffer:SetData failed on chunk " .. i)
            break
        end

        chunks[#chunks + 1] = {
            vb = vb, ib = ib, nv = nv, ni = ni,
            layout = layout, style = style,
            min = Vector3(minX, minY, minZ),
            max = Vector3(maxX, maxY, maxZ),
        }
        totalVerts = totalVerts + nv
        totalTris = totalTris + ni / 3
    end
    f:Close()

    log:Write(LOG_INFO, string.format(
        "[CityMesh] %d chunks, %d verts, %d tris loaded in %.0f ms",
        #chunks, totalVerts, totalTris, (os.clock() - t0) * 1000.0))
    return chunks
end

---Wrap a chunk in a Model. Geometry is set directly rather than going through
---Model:SetVertexBuffers, which is not callable from Lua in this build.
---@param chunk MeshChunk
---@return Model
function CityMesh.MakeModel(chunk)
    local geom = Geometry()
    geom:SetVertexBuffer(0, chunk.vb)
    geom:SetIndexBuffer(chunk.ib)
    -- the fourth argument asks the engine to derive the used vertex range from
    -- the index buffer, which requires shadow data we deliberately do not keep;
    -- the bounding box is set explicitly below instead
    geom:SetDrawRange(TRIANGLE_LIST, 0, chunk.ni, false)

    local model = Model()
    model:SetNumGeometries(1)
    model:SetGeometry(0, 0, geom)
    model:SetGeometryCenter(0, Vector3(0, 0, 0))
    model:SetBoundingBox(BoundingBox(chunk.min, chunk.max))
    return model
end

---Attach every chunk under a parent node with the given material.
---@param parent Node
---@param chunks MeshChunk[]
---@param material Material
---@param castShadows boolean
---@return integer
function CityMesh.Attach(parent, chunks, material, castShadows)
    local n = 0
    for i = 1, #chunks do
        local node = parent:CreateChild("Mesh" .. i)
        local sm = node:CreateComponent("StaticModel")
        sm:SetModel(CityMesh.MakeModel(chunks[i]))
        sm:SetMaterial(material)
        sm.castShadows = castShadows
        n = n + 1
    end
    return n
end

CityMesh.VERTEX_SIZE = VERTEX_SIZE

return CityMesh
