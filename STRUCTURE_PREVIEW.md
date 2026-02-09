# Structure Preview Button

## New Feature: Pre-Analysis Structure Visualization

This feature implements the TODO from `app.py` line 385:
> "Fazer um botao clicavel antes da analise para mostrar a estrutura com as cargas aplicadas, para o usuario ter certeza que esta tudo certo antes de rodar a analise."

### What's New

A new **"👁️ Preview Structure with Loads"** button has been added to the Analysis tab that displays an interactive visualization of the structure before running the FEM analysis.

### Location

The button appears in the **⚙️ Analysis** tab, after the model summary and before the "🚀 Run Analysis" button.

### What It Shows

The preview displays:
- ✅ **Nodes** - Blue markers with IDs and coordinates
- ✅ **Elements** - Black lines connecting nodes
- ✅ **Constraints** - Color-coded boundary condition symbols:
  - 🔴 Red triangle: X-direction fixed
  - 🟢 Green triangle: Y-direction fixed
  - 🟣 Purple circle: Rotation fixed
- ✅ **Point Loads** - Orange arrows showing force direction and magnitude
- ✅ **Distributed Loads** - Multiple arrows along elements
- ✅ **Interactive** - Zoom, pan, and hover for details

### Usage

1. Define your structure in the "📐 Structure Definition" tab
2. Go to the "⚙️ Analysis" tab
3. Click **"👁️ Preview Structure with Loads"**
4. Review the visualization
5. If correct, click **"🚀 Run Analysis"**

### Benefits

- ✅ Catch setup errors before running analysis
- ✅ Visual confirmation of loads and constraints
- ✅ Save time by avoiding incorrect analyses
- ✅ Easy documentation with screenshots

### Implementation

- **Function**: `plot_structure_preview()` in `post_processing/plotter.py`
- **Tests**: `tests/test_structure_preview.py` (5 tests, all passing)
- **Security**: CodeQL scan passed, no vulnerabilities

### Example

For a cantilever beam with a point load:
```
Node 1 (fixed) -------- Node 2 -------- Node 3 (load ↓)
  🔴🟢⚪                                    ⬇️ 1000N
```

The preview shows the nodes, the elements connecting them, the fixed constraint at Node 1 (three symbols for x, y, and rotation), and the downward point load at Node 3.

---
**Status**: ✅ Complete and tested  
**Implemented**: 2025-02-09
