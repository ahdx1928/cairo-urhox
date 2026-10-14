# Cairo · 开罗 — UrhoX 3D 城市浏览器

基于 **Microsoft Global Building Footprints**(2.5 万栋建筑轮廓)+ **OpenStreetMap**(尼罗河河道、路网)离线烘焙,在 UrhoX 引擎(Lua)中渲染的可漫游开罗城。

支持桌面(鼠标/键盘)与手机(摇杆/触控)两种操作方式。

## 目录结构

```
scripts/            游戏运行时代码(Lua)
  main.lua            入口:场景、灯光、材质、HUD、触控
  city/CityMesh.lua   URBXMSH1 二进制网格加载器
  city/CityCamera.lua 环绕相机(桌面鼠标 + 手机手势)
  city/CityTerrain.lua 地形网格挂载
assets/
  City/*.bin          烘焙产物:建筑/河道/地面/路网/地形/海面(~43 MB)
  City/tex_*.png      程序化生成的贴图(立面/屋顶/沙地/岩石)
  Shaders/BLGL/Water.shader  水面波浪 shader(顶点位移 + 菲涅尔)
  Techniques/SkyGradient.xml 渐变天空 technique
city-bake/          离线烘焙管线(Python,不在运行时使用)
  stage1_raw.py       MS 建筑瓦片 -> cairo_raw.bin
  stage3_mesh.py      建筑/河/岛/地面/路网 -> cairo_mesh.bin
  bake_terrain.py     沙漠/山脉/海床/海面 -> cairo_terrain(_mesh)/cairo_sea_mesh.bin
  gen_textures.py     程序化 PNG 贴图
  preview_*.py        软件光栅化预览(无 GPU 环境下的视觉验证)
```

## 运行

用 UrhoX/SCE 编辑器或沙箱打开本目录,入口脚本 `scripts/main.lua`,直接构建预览即可。烘焙产物都在版本库里,克隆下来就能跑,**不需要**重新烘焙。

## 重新烘焙(可选)

只有想改城市范围/层序/风格时才需要。所有路径均为仓库相对路径。

```bash
python3 -m venv --system-site-packages /tmp/venv
/tmp/venv/bin/pip install pyshp numpy pillow

# 1) 数据下载 -> .tmp/cairo/
#    MS 建筑瓦片(Egypt,quadkey 122121122,~95 MB):
#    https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv 中筛选 Egypt 行
curl -L -o .tmp/cairo/egypt_122121122.csv.gz "<tile-url>"
#    OSM 水系 + 路网(Geofabrik,~330 MB zip,只解压两个 shapefile 集):
curl -L -o .tmp/cairo/egypt_shp.zip https://download.geofabrik.de/africa/egypt-latest-free.shp.zip
cd .tmp/cairo && unzip -o egypt_shp.zip "gis_osm_water_a_free_1.*" "gis_osm_roads_free_1.*" && rm egypt_shp.zip

# 2) 烘焙(按顺序)
/tmp/venv/bin/python city-bake/stage1_raw.py       # -> .tmp/cairo/cairo_raw.bin
/tmp/venv/bin/python city-bake/gen_textures.py     # -> assets/City/tex_*.png
/tmp/venv/bin/python city-bake/stage3_mesh.py      # -> assets/City/cairo_mesh.bin
/tmp/venv/bin/python city-bake/bake_terrain.py     # -> cairo_terrain*.bin / cairo_sea_mesh.bin

# 3) 无 GPU 视觉验证
/tmp/venv/bin/python city-bake/preview_textured.py # -> .tmp/cairo/preview_tex.png
/tmp/venv/bin/python city-bake/preview_terrain.py  # -> .tmp/cairo/preview_terrain.png
```

## 关键设计

**高度层序**(所有图层都在两个 bake 脚本里集中定义,改动必须同步核对):

```
海面 -2.40(浪尖 +1.0 也够不到城市)
沙台 -1.20(低于一切城市图层)
地面 -1.00 · 尼罗河 -0.85 · 河心岛 -0.70 · 道路 -0.62(跨河即桥)· 建筑基底 -0.55
```

**网格二进制格式 `URBXMSH1`**:u32 批次数;每批次 7×u32 头(vbytes/ibytes/nv/ni/layout/style/reserved)+ 6×f32 bbox + 顶点字节 + u16 索引。
layout:0=位置+RGBA(16B) 1=位置+UV(20B) 2=位置+法线+UV(32B);style 决定运行时材质(0-3 立面 / 100 屋顶 / 200 河 / 201 岛 / 202 地面 / 203 路 / 300-302 地形沙/岩/海床 / 303 海面)。

**水面**:河与海共用一个 shader,但参数不同——河 `wave_height=0.04`(层间间隙只有 0.15 m,大浪会穿层),海 `0.5`。

**河道三角化必须用耳切**,不能对细长凹多边形用扇形——扇形会扫过所有拐弯内侧,把整片城区铺成水面(曾经的淹城 bug)。

## 数据来源

- 建筑轮廓:[Microsoft Building Footprints](https://github.com/microsoft/UserExperienceRepository) (ODbL)
- 水系/路网/建筑高度缺失时的参照:[OpenStreetMap](https://www.openstreetmap.org) (ODbL)
- 建筑高度数据缺失(`height=-1`),由占地面积 + 市中心密度衰减 + 哈希扰动合成
