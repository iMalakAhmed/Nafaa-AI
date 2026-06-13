# Birth Certificate YOLO Seed Labels

This folder contains seed bounding-box annotations for the 40 labeled birth
certificate images.

Generated files:

- `images/train`, `images/val`: copied certificate images
- `labels/train`, `labels/val`: YOLO-format label files
- `classes.txt`: class names
- `data.yaml`: Ultralytics YOLO dataset config
- `previews`: rendered images with boxes overlaid

Important: these are template-based seed boxes, not final hand-corrected boxes.
The birth certificate layouts are similar, but scans are rotated, cropped, and
shifted, so the boxes must be reviewed and corrected in a labeling tool before
serious YOLO training.

Recommended workflow:

1. Open `data/birthcert_yolo/previews` and spot-check the generated boxes.
2. Import `data/birthcert_yolo` into CVAT, Label Studio, Roboflow, or another
   YOLO-compatible labeler.
3. Correct the boxes against the actual field values.
4. Train YOLO using `data/birthcert_yolo/data.yaml`.

The current corrected annotation pass uses 10 high-value classes:
`child_national_id`, `child_name`, `date_of_birth`, `place_of_birth`,
`father_name`, `mother_name`, `registration_number`, `registration_date`,
`issue_date`, and `serial_number`.
