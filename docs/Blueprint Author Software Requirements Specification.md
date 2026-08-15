# Blueprint Author Software Requirements Specification (SRS)

## Document 1 – Introduction & System Overview

**Version:** 1.0 (Draft)

**Status:** Draft

---

# 1. Introduction

## 1.1 Purpose

The Blueprint Author is a standalone desktop application used to create, edit, validate, and export digital blueprints of shooting targets.

A blueprint serves as the authoritative digital representation of a physical target and contains all geometric, structural, and metadata information required by the runtime scoring system.

The Blueprint Author performs all computationally intensive authoring tasks offline. The exported blueprint is later consumed by the runtime system for registration, bullet localization, and scoring.

The Blueprint Author is intended to eliminate repeated manual setup while ensuring that all runtime scoring operations are based on a validated and consistent reference.

---

## 1.2 Scope

The Blueprint Author shall provide an integrated environment for:

* Importing target images.
* Creating and managing blueprint projects.
* Performing camera-independent calibration.
* Authoring geometric target information.
* Creating scoring zones.
* Defining structural landmarks.
* Extracting repeatable registration features.
* Validating blueprint integrity.
* Exporting runtime-compatible blueprint files.

The application is intended for offline use and is not responsible for bullet detection, scoring, or runtime registration.

---

## 1.3 Objectives

The Blueprint Author shall:

* Produce accurate and repeatable blueprints.
* Minimize operator effort through computer vision assisted authoring.
* Allow manual refinement wherever necessary.
* Preserve physical measurements throughout the authoring process.
* Produce deterministic runtime outputs.
* Support future expansion without breaking previously exported blueprints.

---

# 2. Intended Users

The application is intended for:

* Range operators
* Competition organizers
* System administrators
* Researchers
* Developers creating new target templates

The application is not intended for general end users during live operation.

---

# 3. System Overview

The Blueprint Author is an offline engineering tool responsible for creating the digital reference used throughout the scoring ecosystem.

A blueprint is created once for each target design and may subsequently be reused across any number of physical copies of the same target.

The runtime system does not modify blueprint geometry.

Instead, it aligns incoming images to the stored blueprint before performing scoring.

This establishes the blueprint as the single source of truth for all target geometry.

---

# 4. Application Responsibilities

The Blueprint Author shall be responsible for:

* Project management
* Image import
* Calibration
* Perspective correction
* Physical scale determination
* Paper boundary definition
* Silhouette authoring
* Zone authoring
* Landmark authoring
* Registration feature generation
* Blueprint validation
* Blueprint export

The application shall not perform:

* Bullet detection
* Impact localization
* Score computation
* Camera control
* Live image acquisition

These responsibilities belong to the runtime system.

---

# 5. High-Level Workflow

The Blueprint Author follows a sequential authoring workflow.

```text
Create Project
       │
       ▼
Import Target Image
       │
       ▼
AprilTag Detection
       │
       ▼
Perspective Rectification
       │
       ▼
Physical Scale Calibration
       │
       ▼
Paper Boundary Authoring
       │
       ▼
Silhouette Authoring
       │
       ▼
Zone Authoring
       │
       ▼
Landmark Authoring
       │
       ▼
Registration Feature Generation
       │
       ▼
Validation
       │
       ▼
Blueprint Export
```

Each stage progressively enriches the blueprint with additional validated information.

---

# 6. Design Principles

The Blueprint Author shall adhere to the following principles.

## 6.1 Single Source of Truth

The exported blueprint shall be the authoritative description of a target.

Runtime processing shall never redefine blueprint geometry.

---

## 6.2 Offline Authoring

All authoring operations shall be performed offline.

Runtime systems shall consume exported blueprints without modification.

---

## 6.3 Assisted Authoring

Computer vision algorithms shall assist the operator wherever practical.

The operator shall retain final authority over all blueprint geometry.

---

## 6.4 Physical Consistency

All geometric data shall remain physically meaningful.

Measurements shall be represented in real-world units derived through the calibration process rather than arbitrary image resolution. This ensures that exported blueprints remain independent of the camera used to capture the reference image. 

---

## 6.5 Deterministic Export

Exporting the same validated project shall always produce identical blueprint data.

---

## 6.6 Extensibility

The application shall support future authoring tools and feature types without requiring changes to the runtime blueprint specification.

---

# 7. System Boundary

The Blueprint Author is responsible for creating the blueprint.

The runtime system is responsible for using the blueprint.

```text
                  Blueprint Author
                         │
                         │
          Creates Validated Blueprint
                         │
                         ▼
               target_blueprint.json
                         │
                         ▼
                Runtime Scoring System
                         │
                         ▼
      Registration → Detection → Scoring
```

The interface between both systems shall be the exported blueprint only.

---

# 8. Assumptions

The following assumptions apply:

* The target image is captured under suitable lighting.
* The entire target is visible.
* Calibration markers are visible.
* The reference target is free from significant damage.
* The operator has sufficient permissions to create and modify blueprint projects.

---

# 9. Constraints

The Blueprint Author shall operate entirely as a standalone desktop application.

The application shall:

* Operate without requiring network connectivity.
* Store projects locally.
* Export runtime-compatible blueprint files.
* Preserve project data across software versions whenever practical.

---

# 10. Success Criteria

A blueprint shall be considered complete when:

* All mandatory authoring stages have been completed.
* Validation succeeds without critical errors.
* The blueprint can be exported successfully.
* The exported blueprint is accepted by the runtime system without modification.

---

## Review

I think this is a solid foundation. It clearly establishes the purpose, scope, responsibilities, and guiding principles of the Blueprint Author without tying us to implementation details.

One addition I would make—starting with the next document—is to define **every functional requirement using RFC 2119 terminology** ("shall", "should", "may", "must not"). That gives each requirement a clear level of obligation and makes the specification much easier to verify during implementation and testing.
