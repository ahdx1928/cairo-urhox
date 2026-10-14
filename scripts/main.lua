-- Cairo — a browsable reconstruction of central Cairo built from Microsoft's
-- Global Building Footprints and OpenStreetMap water polygons.
--
-- Every vertex the engine draws is baked offline (see the bake pipeline notes);
-- this script only uploads byte blobs and drives the camera.

local UI = require("urhox-libs/UI")
local SkyUtils = require("urhox-libs.Rendering.SkyUtils")
local CityMesh = require("city.CityMesh")
local CityCamera = require("city.CityCamera")
local CityTerrain = require("city.CityTerrain")
local VirtualControls = require("urhox-libs/UI/VirtualControls")

local MESH_FILE = "cairo_mesh.bin"

-- Anchor and window used by the bake, needed to centre the camera on the data
local ANCHOR_LON = 31.2357
local ANCHOR_LAT = 30.0455
local M_PER_DEG_LAT = 110540.0
local M_PER_DEG_LON = 111320.0 * math.cos(math.rad(ANCHOR_LAT))
local WIN_MIN_LON, WIN_MAX_LON = 31.212, 31.268
local WIN_MIN_LAT, WIN_MAX_LAT = 30.018, 30.066

local BUILDINGS = 25569   -- 112 river-channel footprints removed by the baker
local GROUND_EXTENT = 9000.0

---@type Scene|nil
local scene_ = nil
---@type CityCamera|nil
local cam_ = nil
---@type Widget|nil
local infoLabel_ = nil
---@type Widget|nil
local hintLabel_ = nil
---@type table|nil
local joystick_ = nil
local showHelp_ = true
local stats_ = { chunks = 0, verts = 0, tris = 0, ms = 0 }

---@return number
local function windowCenterX()
    return ((WIN_MIN_LON + WIN_MAX_LON) * 0.5 - ANCHOR_LON) * M_PER_DEG_LON
end

---@return number
local function windowCenterZ()
    return ((WIN_MIN_LAT + WIN_MAX_LAT) * 0.5 - ANCHOR_LAT) * M_PER_DEG_LAT
end

---@param color Color
---@param technique string
---@return Material
local function makeMaterial(color, technique)
    local mat = Material:new()
    mat:SetTechnique(0, cache:GetResource("Technique", technique))
    mat:SetShaderParameter("MatDiffColor", Variant(color))
    -- The sun term is already baked into the vertex colours; a specular term on
    -- top of that just washes the facades out.
    mat:SetShaderParameter("MatSpecColor", Variant(Color(0, 0, 0, 1)))
    return mat
end

---Animated water: wave displacement plus a fresnel tint. Falls back to a flat
---colour if the surface shader is unavailable, so the river never disappears.
---
---The wave sum peaks at 2.0, so the surface swings +/- wave_height*2.0 around
---its baked height. The river sits only 0.15 m above the city ground sheet, so
---its material must use a small wave_height (0.04 -> +/-0.08 m) or the trough
---dips under the ground and the river blinks out. The sea has metres of slack.
---@param waveHeight number
---@param waveSpeed number
---@param waveScale number
---@return Material
local function makeWaterMaterial(waveHeight, waveSpeed, waveScale)
    local mat = Material:new()
    if mat:SetSurfaceShader("Shaders/BLGL/Water.shader") then
        mat:SetShaderParameter("shallow_color", Variant(Color(0.16, 0.38, 0.48, 1)))
        mat:SetShaderParameter("deep_color", Variant(Color(0.05, 0.16, 0.26, 1)))
        mat:SetShaderParameter("sky_color", Variant(Color(0.62, 0.74, 0.86, 1)))
        mat:SetShaderParameter("wave_height", Variant(waveHeight))
        mat:SetShaderParameter("wave_scale", Variant(waveScale))
        mat:SetShaderParameter("wave_speed", Variant(waveSpeed))
        mat:SetShaderParameter("ripple_strength", Variant(0.4))
        log:Write(LOG_INFO, "[Cairo] water surface shader active")
        return mat
    end
    log:Write(LOG_WARNING, "[Cairo] water shader unavailable, flat colour fallback")
    return makeMaterial(Color(0.11, 0.30, 0.38, 1.0), "Techniques/NoTextureUnlitVCol.xml")
end

---Unlit material with a tiled diffuse texture.
---@param texturePath string resource-root relative, e.g. "City/tex_sand.png"
---@param tint Color
---@return Material
local function makeTexturedMaterial(texturePath, tint)
    local mat = makeMaterial(tint, "Techniques/DiffUnlit.xml")
    local tex = cache:GetResource("Texture2D", texturePath)
    if tex then
        mat:SetTexture(TU_DIFFUSE, tex)
    else
        log:Write(LOG_WARNING, "[Cairo] texture not resolved: " .. texturePath)
    end
    return mat
end

---Builds the skybox from a generated gradient cubemap.
---
---Deliberately does not use SkyUtils.CreateGradientSky: its technique
---("Techniques/DiffSkybox.xml") hardcodes a default `<texture unit="0">` pointing
---at Textures/Skybox.ktx, which is not part of this build. Loading it makes the
---engine fetch a 15 MB texture we never use and log "Could not find resource".
---Techniques/SkyGradient.xml is the same pass without that default texture.
---@param scene Scene
---@param opts table
---@return Node
local function createGradientSky(scene, opts)
    local hdrBoost = opts.hdrBoost or 2.0
    local skyTex = SkyUtils.MakeGradientCubemap(opts)

    local mat = Material:new()
    mat:SetTechnique(0, cache:GetResource("Technique", "Techniques/SkyGradient.xml"))
    mat:SetTexture(TU_DIFFUSE, skyTex)
    mat:SetShaderParameter("MatDiffColor", Variant(Vector4(hdrBoost, hdrBoost, hdrBoost, 1.0)))

    local skyNode = scene:CreateChild("Sky")
    local skybox = skyNode:CreateComponent("Skybox")
    skybox:SetModel(cache:GetResource("Model", "Models/Box.mdl"))
    skybox:SetMaterial(mat)
    return skyNode
end

local function createScene()
    scene_ = Scene()
    scene_:CreateComponent("Octree")

    -- late afternoon: sun low in the west, warm light, sandy haze
    local lightGroup = scene_:CreateChild("LightGroup")
    local xml = cache:GetResource("XMLFile", "LightGroup/Dusk.xml")
        or cache:GetResource("XMLFile", "LightGroup/Daytime.xml")
    if xml then
        lightGroup:LoadXML(xml:GetRoot())
    end

    local zone = lightGroup:GetComponent("Zone", true)
    if zone then
        -- Drive ambient from an explicit colour rather than the prebaked SH/IBL.
        -- The gradient sky is deliberately bright, and when it feeds the image
        -- based lighting it floods the whole city with light and washes every
        -- texture out. A fixed fill keeps the result predictable.
        zone.ambientSource = AMBIENT_COLOR
        zone.ambientColor = Color(0.30, 0.33, 0.38, 1.0)
        zone.ambientIntensity = 1.0

        -- The Dusk preset ships a tonemap LUT graded for a dim sunset. Applied to
        -- a bright, evenly lit city it lifts the mid-tones and flattens every
        -- texture toward white. Straight ACES keeps the contrast the textures
        -- actually have.
        zone.tonemapLUTEnabled = false
        zone.tonemapMode = TONEMAP_MODE_ACES
        zone.autoExposureEnabled = false
        zone.vignetteEnabled = false

        -- Bloom is the other half of the washout. The preset's threshold of 0.30
        -- is below the brightness of most of this scene, so nearly every surface
        -- bloomed and the blurred copy smeared the city into a white haze. Only
        -- genuinely blown highlights should glow, and there are none here.
        zone.bloomThreshold = 2.0
        zone.bloomWeight = 0.0
        zone.bloomPlusEnabled = false
        zone.bloomPlusIntensity = 0.0
        zone.bloomUEIntensity = 0.0
        zone.bloomUEThreshold = 2.0

        -- Distance haze only, pushed well past the city so the ground texture
        -- is not veiled. The camera works from ~100 m up to ~4 km out.
        zone.heightFog = false
        zone.fogColor = Color(0.72, 0.66, 0.56, 1.0)
        zone.fogStart = 6000.0
        zone.fogEnd = 20000.0
        zone.fogDensity = 1.0
        zone.boundingBox = BoundingBox(Vector3(-20000, -800, -20000), Vector3(20000, 2500, 20000))
        zone.override = true
    end

    local light = lightGroup:GetComponent("Light", true)
    if light then
        -- Deliberately high-contrast: the Dusk preset's 5.0 assumes auto-exposure
        -- and, with the sky also feeding ambient, was flattening every surface
        -- toward white. These values keep sunlit faces well clear of clipping.
        light.color = Color(1.0, 0.92, 0.80)
        light.brightness = 0.60
        light.castShadows = true
        light.shadowResolution = 2048
        light.shadowCascade = CascadeParameters(60.0, 260.0, 900.0, 0.0, 0.85)
    end

    createGradientSky(scene_, {
        zenith = Color(0.10, 0.22, 0.46),
        horizon = Color(0.78, 0.70, 0.58),
        ground = Color(0.40, 0.34, 0.26),
        skyExp = 0.45,
        -- the sky is also an HDR source for reflections, so keep it near 1.0 or
        -- it feeds extra light back into the scene
        hdrBoost = 1.0,
    })
end

---@return boolean
local function buildCity()
    local t0 = os.clock()
    local chunks = CityMesh.Load(MESH_FILE)
    if #chunks == 0 then
        log:Write(LOG_ERROR, "[Cairo] no mesh chunks loaded")
        return false
    end

    local root = scene_:CreateChild("Cairo")

    -- Facade textures, indexed to match the baker's style enum. Each gets a
    -- distinct tint so neighbouring blocks stay visually separated even when
    -- distant facades mip down to an average colour.
    local FACADES = {
        { path = "City/tex_wall_a.png", tint = Color(1.00, 0.97, 0.90, 1.0) }, -- plaster
        { path = "City/tex_wall_b.png", tint = Color(1.00, 0.92, 0.86, 1.0) }, -- brick
        { path = "City/tex_wall_c.png", tint = Color(0.94, 0.95, 0.97, 1.0) }, -- concrete
        { path = "City/tex_wall_d.png", tint = Color(0.82, 0.90, 1.00, 1.0) }, -- glass
    }
    local wallMats = {}
    for i = 1, #FACADES do
        wallMats[i] = makeTexturedMaterial(FACADES[i].path, FACADES[i].tint)
    end
    local matRoof = makeTexturedMaterial("City/tex_roof.png", Color(1, 1, 1, 1))
    local matGround = makeTexturedMaterial("City/tex_sand.png", Color(1, 1, 1, 1))
    local matIsland = makeTexturedMaterial("City/tex_sand_island.png", Color(1, 1, 1, 1))
    -- River: tight waves so the surface never swings under the ground sheet.
    local matWater = makeWaterMaterial(0.04, 1.1, 0.05)
    -- roads and footways carry their colour in the vertices
    local matRoad = makeMaterial(Color(1, 1, 1, 1), "Techniques/NoTextureUnlitVCol.xml")

    -- chunk tags written by the baker; see city-bake/stage3_mesh.py
    local ROOF_STYLE = 100
    local STYLE_ISLAND = 201
    local STYLE_GROUND = 202
    local STYLE_ROAD = 203

    for i = 1, #chunks do
        local chunk = chunks[i]
        local mat
        if chunk.style == ROOF_STYLE then
            mat = matRoof
        elseif chunk.style == STYLE_ISLAND then
            mat = matIsland
        elseif chunk.style == STYLE_GROUND then
            mat = matGround
        elseif chunk.style == STYLE_ROAD then
            mat = matRoad
        elseif chunk.layout == 2 then
            mat = wallMats[(chunk.style % #wallMats) + 1]
        else
            mat = matWater
        end
        local node = root:CreateChild("Mesh" .. i)
        local sm = node:CreateComponent("StaticModel")
        sm:SetModel(CityMesh.MakeModel(chunk))
        sm:SetMaterial(mat)
        sm.castShadows = false
    end

    local verts, tris = 0, 0
    for i = 1, #chunks do
        verts = verts + chunks[i].nv
        tris = tris + chunks[i].ni / 3
    end
    -- surrounding desert, mountains and sea
    local terrainParent = assert(scene_, "scene missing")
    -- the sea is a flat plane at sea level; the terrain only supplies the seabed.
    -- The sea has metres of clearance under the wave crests, so it can afford
    -- the big rolling swell the river cannot.
    local seaChunks = CityMesh.Load("cairo_sea_mesh.bin")
    if #seaChunks > 0 then
        local matSea = makeWaterMaterial(0.5, 0.6, 0.012)
        CityTerrain.Attach(terrainParent, seaChunks, { [303] = matSea })
        log:Write(LOG_INFO, "[Cairo] sea surface attached")
    end
    CityTerrain.Build(terrainParent, CityMesh.Load, {
        [CityTerrain.STYLE_SAND] = makeTexturedMaterial("City/tex_desert.png", Color(1, 1, 1, 1)),
        [CityTerrain.STYLE_ROCK] = makeTexturedMaterial("City/tex_rock.png", Color(1, 1, 1, 1)),
        -- Seabed. The animated surface above it comes from cairo_sea_mesh.bin, so
        -- this layer is just dark ground; giving it the water material too would
        -- stack two wave surfaces on top of each other.
        [CityTerrain.STYLE_WATER] = makeMaterial(Color(0.09, 0.19, 0.23, 1.0),
            "Techniques/NoTextureUnlitVCol.xml"),
    })

    stats_.chunks = #chunks
    stats_.verts = verts
    stats_.tris = tris
    stats_.ms = (os.clock() - t0) * 1000.0
    log:Write(LOG_INFO, string.format("[Cairo] attached %d chunks in %.0f ms", #chunks, stats_.ms))
    return true
end

---Whether to build the on-screen controls.
---
---A browser on a phone reports platform "Web", not "Android"/"iOS", so keying
---off the platform name alone silently disabled every control there. Treat
---anything that can receive touches as touch-capable, and let desktop fall back
---to mouse emulation so the same controls can be driven with a mouse.
---@return boolean
local function wantsOnScreenControls()
    local p = GetPlatform()
    if p == "Android" or p == "iOS" or p == "HarmonyOS" then
        return true
    end
    if p == "Web" then
        return true
    end
    -- last resort: the engine exposes a touch count on touch-capable builds
    return input ~= nil and input.numTouches ~= nil
end

local function createHUD()
    local touch = wantsOnScreenControls()
    infoLabel_ = UI.Label { text = "", fontSize = 13, color = "#e9dcc2" }
    hintLabel_ = UI.Label {
        text = touch
            and "单指拖动 旋转 · 双指拖动 平移 · 双指捏合 缩放"
            or "左键拖动 旋转 · 右键拖动 平移 · 滚轮 缩放 · WASD/QE 移动 · R 复位 · H 隐藏",
        fontSize = 12,
        color = "#a99c85",
    }

    -- The corner card. Its size comes from its content; the absolute offsets
    -- only mean something once it has a parent to sit inside.
    -- pointerEvents="none" is important on phones: otherwise a drag that starts
    -- on the card is swallowed by it instead of rotating the camera.
    local card = UI.Panel {
        position = "absolute",
        left = touch and 10 or 18,
        bottom = touch and 10 or 18,
        padding = touch and 10 or 14,
        borderRadius = 10,
        backgroundColor = "#12151adf",
        pointerEvents = "none",
        flexDirection = "column",
        children = {
            UI.Label { text = "开罗 · CAIRO", fontSize = touch and 16 or 20, color = "#f7d9a0" },
            UI.Label {
                text = "尼罗河两岸 · 扎马雷克岛 · 解放广场",
                fontSize = touch and 10 or 12,
                color = "#9c9384",
            },
            infoLabel_,
            UI.Divider {},
            hintLabel_,
        },
    }

    -- Yoga sizes the root to the constraints it is given, so the root has to be
    -- a full-screen transparent wrapper. Making the card itself the root pins it
    -- to the whole screen, and its translucent background then hides the scene.
    local root = UI.Panel {
        width = "100%",
        height = "100%",
        pointerEvents = "box-none",
        children = { card },
    }
    UI.SetRoot(root)

    infoLabel_:SetText(string.format(
        "v5 · %d 栋建筑 · %d 万顶点 · %d 批次",
        BUILDINGS, math.floor(stats_.verts / 10000), stats_.chunks))
end

-- ============================================================================
-- On-screen controls (mobile)
--
-- VirtualControls owns the touch handling, DPI scaling and rendering; this
-- only maps its output onto the camera. It also draws the stick, which is the
-- part a raw TouchBegin subscription cannot give you.
-- ============================================================================
local function setupTouchControls()
    if not wantsOnScreenControls() then
        log:Write(LOG_INFO, "[Cairo] no touch support: mouse + keyboard only")
        return
    end

    -- VirtualControls gates its touch handling on being "mobile". Turning on
    -- the engine's touch emulation satisfies that check on Web and desktop, and
    -- the mouse emulation below lets a mouse drive the same controls.
    input:SetTouchEmulation(true)
    VirtualControls.SetMouseEmulation(true)

    -- one-finger drag anywhere on the right half turns the view
    VirtualControls.CreateTouchLookArea({
        regionPreset = "full_screen",
        sensitivity = 0.15,
        on_look = function(dYaw, dPitch)
            if cam_ then
                cam_:Look(dYaw, dPitch)
            end
        end,
    })

    -- left thumb stick pans the focus point across the city
    joystick_ = VirtualControls.CreateJoystick({
        position = Vector2(150, -150),
        alignment = { HA_LEFT, VA_BOTTOM },
        baseRadius = 80,
        knobRadius = 32,
        deadZone = 0.12,
        opacity = 0.45,
        activeOpacity = 0.8,
        keyBinding = "WASD",
        on_move = function(x, y, percent)
            if cam_ then
                cam_:Pan(x, y, percent)
            end
        end,
    })

    log:Write(LOG_INFO, "[Cairo] mobile controls: joystick + touch look area")
end

function Start()
    graphics.windowTitle = "Cairo"

    -- Must happen before anything reads input: VirtualControls decides whether
    -- it is on a touch device by checking input.touchEmulation, and a browser
    -- reports platform "Web" rather than a mobile name, so without this the
    -- on-screen controls refuse every touch.
    if wantsOnScreenControls() then
        input:SetTouchEmulation(true)
    end

    createScene()

    if not buildCity() then
        log:Write(LOG_ERROR, "[Cairo] city build failed")
    end

    cam_ = CityCamera.Create(scene_, {
        target = Vector3(windowCenterX(), 30.0, windowCenterZ()),
        yaw = -0.55,
        pitch = -0.60,
        -- Close enough that a 7 m window bay covers several screen pixels.
        -- From 1.5 km it was under one pixel, so every facade mip-mapped down
        -- to a flat average and the city read as untextured.
        dist = 420.0,
        fov = 52.0,
    })
    renderer:SetViewport(0, Viewport:new(scene_, cam_.camera))
    renderer.hdrRendering = true
    -- Facades are seen at glancing angles across the whole city, so anisotropic
    -- filtering is what keeps the window pattern from blurring away.
    renderer.textureAnisotropy = 8
    renderer.textureFilterMode = FILTER_ANISOTROPIC

    -- Strip the preset's post-processing chain. The Dusk render path runs a
    -- bloom with a 0.30 threshold, which is below the brightness of most of
    -- this scene, so nearly every surface bloomed and the blurred copy smeared
    -- the city into white. Auto-exposure and depth-of-field are equally
    -- unwanted on a fixed-lighting aerial view.
    local vp = renderer:GetViewport(0)
    local rp = vp and vp.renderPath
    if rp then
        for _, tag in ipairs({
            "BloomHDR Downsample", "BloomHDR Upsample",
            "BloomHDRPlus", "BloomHDRUE", "GaussianBlur",
            "AutoExposure", "DOF",
        }) do
            if rp:IsAdded(tag) then
                rp:SetEnabled(tag, false)
            end
        end
        log:Write(LOG_INFO, "[Cairo] post-processing disabled: bloom/exposure/dof")
    end

    UI.Init({ theme = "default-dark", scale = UI.Scale.DEFAULT })
    createHUD()

    SubscribeToEvent("Update", "HandleUpdate")
    setupTouchControls()
    log:Write(LOG_INFO, "[Cairo] ready")
end

function Stop()
    UI.Shutdown()
end

---@param eventType string
---@param eventData UpdateEventData
function HandleUpdate(eventType, eventData)
    local dt = eventData["TimeStep"]:GetFloat()
    if cam_ then
        cam_:Update(dt)
    end
    if input:GetKeyPress(KEY_H) then
        showHelp_ = not showHelp_
        if hintLabel_ then
            hintLabel_:SetVisible(showHelp_)
        end
    end
end
