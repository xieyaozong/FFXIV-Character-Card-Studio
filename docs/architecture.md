# Architecture

```text
Screenshot import
  -> image validation
  -> background removal and review crops
  -> palette extraction
  -> VLM evidence candidates
  -> multilingual entity resolution
  -> user confirmation
  -> character / outfit profiles
  -> prompt plans
  -> diffusion panels
  -> deterministic card layout
```

The VLM never writes directly to a final profile. It returns evidence-linked candidates that remain editable. Jobs, weapons, pets, mounts, and props are optional and stay absent unless visible or explicitly added by the user.
