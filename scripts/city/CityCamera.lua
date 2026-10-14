-- Orbit camera for browsing the city.
--
--   left drag / arrow keys   orbit
--   right or middle drag     pan
--   wheel                    zoom
--   WASD, Q/E                pan and raise the focus point
--   R                        reset the view
--
-- Pitch is clamped just short of vertical so the up vector never degenerates.

local CityCamera = {}

local MIN_PITCH = -1.45
local MAX_PITCH = -0.02
local MIN_DIST = 25.0
local MAX_DIST = 22000.0

---@class CityCamera
---@field node Node
---@field camera Camera
---@field target Vector3
---@field yaw number
---@field pitch number
---@field dist number
---@field home table
local CityCamera_ = {}
CityCamera_.__index = CityCamera_

---@param scene Scene
---@param opts? table
---@return CityCamera
function CityCamera.Create(scene, opts)
    opts = opts or {}
    local home = {
        target = opts.target or Vector3(0, 0, 0),
        yaw = opts.yaw or -0.55,
        pitch = opts.pitch or -0.62,
        dist = opts.dist or 1400.0,
        fov = opts.fov or 52.0,
    }

    local node = scene:CreateChild("Camera")
    local cam = node:CreateComponent("Camera")
    cam.fov = home.fov
    cam.nearClip = 6.0
    -- The terrain reaches out ~31 km, so the far plane has to clear that or
    -- the mountains get clipped away. The near plane is pushed out to match,
    -- which keeps enough depth precision to avoid z-fighting on the city.
    cam.farClip = 42000.0

    ---@type CityCamera
    local self = setmetatable({}, CityCamera_)
    self.node = node
    self.camera = cam
    self.target = Vector3(home.target.x, home.target.y, home.target.z)
    self.yaw = home.yaw
    self.pitch = home.pitch
    self.dist = home.dist
    self.home = home
    self:_apply()
    return self
end

function CityCamera_:_apply()
    local cp = math.cos(self.pitch)
    local sp = math.sin(self.pitch)
    local cy = math.cos(self.yaw)
    local sy = math.sin(self.yaw)
    local x = self.target.x - self.dist * cp * sy
    local y = self.target.y - self.dist * sp
    local z = self.target.z - self.dist * cp * cy
    self.node:SetPosition(Vector3(x, y, z))
    self.node:LookAt(self.target, Vector3.UP)
end

function CityCamera_:Reset()
    self.target = Vector3(self.home.target.x, self.home.target.y, self.home.target.z)
    self.yaw = self.home.yaw
    self.pitch = self.home.pitch
    self.dist = self.home.dist
    self:_apply()
end

-- ============================================================================
-- Gesture API
--
-- The mobile controls are driven by urhox-libs VirtualControls:
--   * CreateTouchLookArea feeds Look(dYaw, dPitch) for one-finger drags
--   * CreateJoystick feeds Pan(x, y, percent) for the on-screen stick
-- Keeping the maths here means both the library and the desktop mouse path
-- share one implementation.
-- ============================================================================

---Orbit from a look-area drag. Deltas are already in logical pixels.
---@param dYaw number
---@param dPitch number
function CityCamera_:Look(dYaw, dPitch)
    self.yaw = self.yaw - dYaw * 0.006
    self.pitch = math.max(MIN_PITCH, math.min(MAX_PITCH, self.pitch - dPitch * 0.006))
    self:_apply()
end

---Zoom by a multiplicative factor (>1 pulls back, <1 moves in).
---@param factor number
function CityCamera_:Zoom(factor)
    self.dist = math.max(MIN_DIST, math.min(MAX_DIST, self.dist * factor))
    self:_apply()
end

---Move the focus point from a joystick. x/y are -1..1 in screen space:
---x to the right, y up (the library reports up as positive).
---@param x number
---@param y number
---@param percent number
function CityCamera_:Pan(x, y, percent)
    local mag = math.sqrt(x * x + y * y)
    if mag < 1e-4 then
        return
    end
    local k = self.dist * 0.02 * percent
    -- joystick right => move the target right; joystick up => move it forward
    local cy = math.cos(self.yaw)
    local sy = math.sin(self.yaw)
    local fx, fz = sy, cy      -- camera forward projected on the ground
    local rx, rz = cy, -sy     -- camera right
    self.target = Vector3(
        self.target.x + (rx * x + fx * y) * k,
        self.target.y,
        self.target.z + (rz * x + fz * y) * k)
    self:_apply()
end

---@param dt number
function CityCamera_:Update(dt)
    local dx = input.mouseMoveX
    local dy = input.mouseMoveY
    local orbiting = input:GetMouseButtonDown(MOUSEB_LEFT)
    local panning = input:GetMouseButtonDown(MOUSEB_RIGHT)
        or input:GetMouseButtonDown(MOUSEB_MIDDLE)

    if orbiting then
        self.yaw = self.yaw - dx * 0.005
        self.pitch = math.max(MIN_PITCH, math.min(MAX_PITCH, self.pitch - dy * 0.005))
    elseif panning then
        -- pan in the camera's ground plane; scale with distance so it feels
        -- the same whether you are at street level or over the whole city
        local k = self.dist * 0.0016
        local cy = math.cos(self.yaw)
        local sy = math.sin(self.yaw)
        -- camera right = (cos(yaw), 0, -sin(yaw)); forward on ground = (sin, 0, cos)
        self.target = Vector3(
            self.target.x - (dx * cy - dy * sy) * k,
            self.target.y,
            self.target.z + (dx * sy + dy * cy) * k)
    end

    if input:GetKeyDown(KEY_LEFT) then self.yaw = self.yaw + dt * 0.9 end
    if input:GetKeyDown(KEY_RIGHT) then self.yaw = self.yaw - dt * 0.9 end
    if input:GetKeyDown(KEY_UP) then
        self.pitch = math.min(MAX_PITCH, self.pitch + dt * 0.6)
    end
    if input:GetKeyDown(KEY_DOWN) then
        self.pitch = math.max(MIN_PITCH, self.pitch - dt * 0.6)
    end

    local wheel = input.mouseMoveWheel
    if wheel ~= 0 then
        self.dist = math.max(MIN_DIST, math.min(MAX_DIST, self.dist * (1.0 - wheel * 0.12)))
    end

    -- keyboard pan / lift
    local k = self.dist * dt * 0.55
    local cy = math.cos(self.yaw)
    local sy = math.sin(self.yaw)
    local fwdX, fwdZ = sy, cy
    local rgtX, rgtZ = cy, -sy
    if input:GetKeyDown(KEY_W) then
        self.target = Vector3(self.target.x + fwdX * k, self.target.y, self.target.z + fwdZ * k)
    end
    if input:GetKeyDown(KEY_S) then
        self.target = Vector3(self.target.x - fwdX * k, self.target.y, self.target.z - fwdZ * k)
    end
    if input:GetKeyDown(KEY_A) then
        self.target = Vector3(self.target.x - rgtX * k, self.target.y, self.target.z - rgtZ * k)
    end
    if input:GetKeyDown(KEY_D) then
        self.target = Vector3(self.target.x + rgtX * k, self.target.y, self.target.z + rgtZ * k)
    end
    if input:GetKeyDown(KEY_E) then
        self.target = Vector3(self.target.x, self.target.y + k, self.target.z)
    end
    if input:GetKeyDown(KEY_Q) then
        self.target = Vector3(self.target.x, self.target.y - k, self.target.z)
    end
    if input:GetKeyPress(KEY_R) then
        self:Reset()
    end

    self:_apply()
end

return CityCamera
