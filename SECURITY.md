# Security policy

## Report a vulnerability

Do not open a public issue for a security problem. Use GitHub's private vulnerability reporting:

1. Open the repo's **Security** tab.
2. Click **Report a vulnerability**.
3. Tell us what the problem is, how to make it happen, and which version you used (`recapture --version`). Attach a page, box file or config that shows it, if you can.

Only you and the maintainers can see the report. We will reply in the report and tell you when a fix is out.

## What counts

Recapture reads pages, box files and configs, and it downloads assets. A security problem is a way to make it do more than that. For example:

- A crafted image, box file or config that makes Recapture run code.
- A config, asset path or box file that makes `render`, `preview`, `export` or `init` write or delete files outside the output folder you gave.
- A crafted asset archive that makes `fetch-assets` write files outside the asset folder.

A slow render, high memory use or a crash on a broken input is a bug. Report it with the bug report template.

## Supported versions

We fix security problems in the latest release only. Update to it before you report, and check that the problem is still there.
