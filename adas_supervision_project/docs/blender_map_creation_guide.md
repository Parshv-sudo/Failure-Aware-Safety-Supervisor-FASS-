# Free Custom Map Creation Pipeline for CARLA Base (Blender + Blosm)

Since RoadRunner is expensive, the standard free alternative used by researchers and students involves using **Blender** with the **Blosm (Blender-OSM)** add-on to get 3D Google Earth maps and OpenStreetMap logic.

This pipeline will give you photorealistic buildings and terrain (for perception models) seamlessly aligned with OpenDRIVE logic (for ADAS logic and CARLA's Traffic Manager).

---

## 🛠 Prerequisites

1.  **Blender (Free):** Download and install Blender (version 3.x is usually safest for CARLA plugins).
2.  **Blosm Add-on (Base Version is Free):** Download from gumroad (Blender-OSM). It imports 3D buildings, terrain, and satellite imagery into Blender.
3.  **CARLA UE4 Source Build:** You **MUST** have the Unreal Engine 4.26 editor version of CARLA (the source build, not just the `.zip` / `.tar.gz` precompiled version). Custom 3D map meshes cannot be dragged and dropped directly into Python; the map must be compiled ("cooked") first.

---

## Stage 1: Get the 3D Visual Mesh in Blender

1.  **Install Blosm in Blender**:
    *   Open Blender → Edit → Preferences → Add-ons → Install (Select the downloaded Blosm `.zip`).
    *   Activate it and set the required API keys (Mapbox token for satellite imagery / terrain if asked).
2.  **Select Area**:
    *   Open the Blosm panel (press `N` in the Blender viewport).
    *   Click **Select** to open a map in your browser. Choose the **exact same bounding area** (e.g., your NYC intersection).
    *   Click **Copy Coordinates**.
3.  **Import 3D Mesh**:
    *   In Blender, paste the coordinates.
    *   Under Import, choose **Google 3D Tiles** (if using Blosm premium) OR choose **OSM > 3D Buildings + Roads + Terrain** (for the free version paired with Mapbox satellite texture).
    *   *Crucial Step:* Make sure you note down the exact Latitude/Longitude of the **Origin Point**. Every object in this map MUST share the same (0, 0, 0) origin center as your CARLA OpenDRIVE file.
4.  **Export Mesh**:
    *   Clean up any overlapping or floating meshes.
    *   File → Export → **FBX (`.fbx`)**. Uncheck "Bake Animation" and ensure scale is set to `1.0`.

---

## Stage 2: Generate the Logic Network (`.xodr`)

Your 3D map looks pretty, but cars will just crash into it because they don't know where the lanes are.

1.  Use your `fetch_osm_map.py` script to pull the `.osm` file for the **exact same coordinates** you used in Blosm.
2.  Use CARLA's built-in `Osm2Odr` python API to convert the `.osm` into `.xodr` (OpenDRIVE).
    *   You can set the `offset_x` and `offset_y` parameters in CARLA's conversion script so it perfectly matches your Blender origin.

---

## Stage 3: The Fusion (Unreal Engine 4 & CARLA Source)

This is where the magic happens. You overlay the invisible road logic onto your visible 3D mesh.

1.  **Create a New Map in UE4**:
    *   Open the CARLA UE4 Editor.
    *   Go to `Content/Carla/Maps` and create a blank Level (e.g., `NYC_Accidents_Map`).
2.  **Import the 3D World**:
    *   Drag your `.fbx` file from Blender into the UE4 Content Browser.
    *   Drag it into the scene. Make sure its Location Transform is exactly `(X:0, Y:0, Z:0)`.
3.  **Import the OpenDRIVE Logic (`.xodr`)**:
    *   Put your generated `.xodr` file into the `Content/Carla/Maps/OpenDrive` folder in UE4.
    *   In the UE4 Editor, use CARLA's **OpenDRIVE generator utility** to spawn the road splines in the map.
4.  **Align and Fine Tune**:
    *   Look at your map. The `.xodr` spline roads should perfectly overlap your Google/Blender visual roads.
    *   If they are slightly off, shift the entire `.fbx` mesh until the visual lanes are directly underneath the invisible OpenDRIVE spline nodes.
    *   Once perfectly aligned, ensure the CARLA generated road surface mesh is set to **Invisible (`Visibility: False`)**, but ensure its **Collision is turned ON**.
5.  **Compile (Cook) the Map**:
    *   Run `make package` or `make package ARGS="--packages=YourPackage"` in your CARLA root terminal.
    *   This generates `.pak` files that you can now load using your `start_fass_autonomous.bat` and `main.py --map` python scripts.

---

## 🧠 Why this is perfect for your ADAS Project:

During simulation:
*   Your perception ML model sees the photorealistic Google 3D buildings and trees.
*   Your ADAS `takeover_logic` and risk calculators (`TTC`, `THW`) compute distances exactly off the `.xodr` semantic lane boundaries.
*   When you deliberately spawn "accident" scenarios (e.g., a pedestrian jaywalking or car running a red light), they follow the OpenDRIVE boundaries, but the dashboard cameras record photorealistic chaos.
