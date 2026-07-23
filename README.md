# Target-Blueprint-Authoring-System
This is a subsystem of a private project that i am working on that is used to detect and score bullet holes on a target board in outdoor conditions

## Objective.

A blueprint is a complete digital representation of a shooting target that contains:

* Reference image metadata
* Physical scale calibration
* AprilTag configuration
* Coordinate system
* Scoring zones
* Structural landmarks
* Paper boundary
* Validation information

The generated blueprint becomes the authoritative source used by the runtime scoring system. The runtime should never require manual authoring again.

---

# Overall Workflow

The project represents an offline authoring pipeline.

```
Reference Image
        │
        ▼
Image Loading
        │
        ▼
Image Preprocessing
        │
        ▼
AprilTag Detection
        │
        ▼
Reference Registration
        │
        ▼
AprilTag Geometry Analysis
        │
        ▼
Physical Scale Calibration
        │
        ▼
Blueprint Creation
        │
        ▼
Interactive Zone Authoring
        │
        ▼
Landmark Authoring
        │
        ▼
Paper Boundary Authoring
        │
        ▼
Blueprint Validation
        │
        ▼
Blueprint Export
```

Everything after export belongs to the runtime application and is outside the scope of this project.

---

# Design Philosophy

The project is designed around several core principles.

## One-Time Authoring

Every target template should only be authored once.

Examples:

* IPSC silhouette
* NSRA targets
* ISSF targets
* Police qualification targets
* Custom departmental targets

Once authored, no manual work should ever be required again.

---

## Human-in-the-loop

Unlike runtime processing, authoring prioritizes correctness over speed.

The operator should manually verify:

* zone placement
* calibration
* landmarks
* paper contour

Every authored element should be visually confirmed before export.

---

## Deterministic Output

Given the same reference image, the project should always generate an identical blueprint.

No random processes should affect the exported data.

---

## Modular Architecture

Every stage should produce a complete output consumed by the following stage.

Example

```
AprilTag Detection

↓

Geometry Analysis

↓

Scale Calibration

↓

Blueprint Metadata

↓

Zone Authoring
```

No later stage should recompute earlier information.

---

# Reference Image

The reference image is the master representation of the target.

It defines

* image dimensions
* coordinate system
* AprilTag locations
* blueprint origin

Every future runtime image will ultimately be transformed into this coordinate system.

---

# Coordinate System

The blueprint uses image coordinates.

Origin

```
(0,0)
```

Upper-left corner.

Positive X

→ right

Positive Y

↓

downwards

Every stored polygon uses this coordinate system.

---

# AprilTag Detection

The project detects all required AprilTags.

Each tag provides

* ID
* corner locations
* center
* geometry

The project verifies that every mandatory tag exists before continuing.

Missing tags immediately terminate authoring.

---

# Geometry Analysis

Each detected AprilTag is analyzed independently.

Measurements include

Average edge length

Edge consistency

Aspect ratio

Internal angles

Polygon area

Physical scale estimate

These measurements are stored for later calibration.

---

# Physical Scale Calibration

The project derives pixels-per-millimeter directly from AprilTag geometry.

Each tag contributes a scale estimate.

Each estimate receives a quality score based on

* edge consistency
* aspect ratio
* angular consistency

These quality values become weights.

The final scale is

```
Weighted Mean(px/mm)
```

The blueprint stores

```
pixels_per_mm

mm_per_pixel

tag quality

individual estimates
```

for complete traceability.

---

# Blueprint Metadata

Once calibration is complete, the project initializes the blueprint.

This object becomes the central data structure for all subsequent authoring.

It should contain

```
Reference metadata

Scale

AprilTags

Coordinate system

Zones

Landmarks

Paper contour
```

Every later stage modifies this single object.

---

# Interactive Authoring Interface

The remainder of the project transitions from a computer vision pipeline into a lightweight graphical authoring application.

The project should provide an embedded interface instead of relying on terminal input.

The interface should remain entirely inside Jupyter/Colab.

---

# Authoring State

The interface maintains an internal authoring session.

It tracks

Current zone

Current polygon

Completed zones

Current landmark

Current paper contour

Blueprint object

Undo history

Display state

The interface should never rely on global variables.

A dedicated controller class should own all authoring state.

---

# Zone Authoring

The operator manually defines scoring regions.

Each zone is represented as a closed polygon.

Each polygon contains

```
id

label

type

score

polygon vertices
```

Vertices are collected through mouse interaction.

The polygon remains editable until confirmed.

---

# Automatic Zone Progression

The project should not ask the operator which zone to author.

Instead, the template defines an ordered sequence.

Example

```
5

↓

4

↓

3

↓

2

↓

1
```

The interface automatically advances after each confirmed polygon.

---

# Live Visualization

During authoring, the canvas continuously updates.

It displays

Current polygon

Completed polygons

Vertex markers

Polygon edges

Zone labels

Semi-transparent fills

This allows immediate visual verification.

---

# Editing Operations

The interface should support

Undo last vertex

Clear current polygon

Restart current zone

Delete completed zone

Re-edit previous zone

Skip optional zone

The operator should never need to restart the project because of a single mistake.

---

# Landmark Authoring

Once scoring zones are complete, the operator defines structural landmarks.

Landmarks are not scoring elements.

They exist to support future fine registration.

Examples

Head outline

Shoulder points

Torso outline

Waist contour

These features remain relatively stable even after repeated bullet impacts.

---

# Paper Boundary

The final manually authored geometry is the outer paper boundary.

This contour defines

Valid shooting region

Cropping limits

Registration constraints

Visualization boundary

It is stored as a polygon.

---

# Validation

Before export, the project performs consistency checks.

Validation includes

Reference image loaded

AprilTags detected

Scale available

Coordinate system initialized

All mandatory zones authored

Unique IDs

Minimum polygon vertices

Polygon inside image

Landmarks present

Paper contour valid

The project should produce a validation report summarizing all checks.

Export should only be allowed after successful validation.

---

# Blueprint Export

The project exports a JSON blueprint.

The exported file becomes the authoritative representation of the target.

It should be independent of the project.

The runtime application should require only

```
Reference image

Blueprint JSON
```

to operate.

---

# Future Runtime Compatibility

Although this project does not implement runtime processing, every design decision should support downstream use.

The exported blueprint should be usable for:

* Loading target templates.
* Performing AprilTag-based registration.
* Applying physical scale calibration.
* Mapping detections into blueprint coordinates.
* Determining score regions from polygon membership.
* Using landmarks for fine registration.
* Validating detections against the paper boundary.

No additional manual configuration should be required once the blueprint has been authored.

---

## Recommended Software Architecture

Rather than continuing with isolated project cells and global variables, I recommend implementing the remainder of the project around a single `BlueprintAuthor` class. This class should own the blueprint, the authoring state, the interactive canvas, the UI widgets, validation, and export functionality. Each project cell would then either instantiate the class or invoke one of its high-level methods, keeping the code organized, testable, and much closer to the architecture of the eventual desktop application.
