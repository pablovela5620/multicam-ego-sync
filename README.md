# multicam-ego-sync
App to synchronize multicam angles from final cut cam/pro with egocentric video

Uses [Rerun](https://rerun.io/) to visualize, [Gradio](https://www.gradio.app) for an interactive UI, and [Pixi](https://pixi.sh/latest/) for a easy installation

<p align="center">
  <a title="Website" href="https://rerun.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
          <img src="https://img.shields.io/badge/Rerun-0.23.0-blue.svg?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzQ0MV8xMTAzOCkiPgo8cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHJ4PSI4IiBmaWxsPSJibGFjayIvPgo8cGF0aCBkPSJNMy41OTcwMSA1Ljg5NTM0TDkuNTQyOTEgMi41MjM1OUw4Ljg3ODg2IDIuMTQ3MDVMMi45MzMgNS41MTg3NUwyLjkzMjk1IDExLjI5TDMuNTk2NDIgMTEuNjY2MkwzLjU5NzAxIDUuODk1MzRaTTUuMDExMjkgNi42OTc1NEw5LjU0NTc1IDQuMTI2MDlMOS41NDU4NCA0Ljk3NzA3TDUuNzYxNDMgNy4xMjI5OVYxMi44OTM4SDcuMDg5MzZMNi40MjU1MSAxMi41MTczVjExLjY2Nkw4LjU5MDY4IDEyLjg5MzhIOS45MTc5NUw2LjQyNTQxIDEwLjkxMzNWMTAuMDYyMUwxMS40MTkyIDEyLjg5MzhIMTIuNzQ2M0wxMC41ODQ5IDExLjY2ODJMMTMuMDM4MyAxMC4yNzY3VjQuNTA1NTlMMTIuMzc0OCA0LjEyOTQ0TDEyLjM3NDMgOS45MDAyOEw5LjkyMDkyIDExLjI5MTVMOS4xNzA0IDEwLjg2NTlMMTEuNjI0IDkuNDc0NTRWMy43MDM2OUwxMC45NjAyIDMuMzI3MjRMMTAuOTYwMSA5LjA5ODA2TDguNTA2MyAxMC40ODk0TDcuNzU2MDEgMTAuMDY0TDEwLjIwOTggOC42NzI1MlYyLjk5NjU2TDQuMzQ3MjMgNi4zMjEwOUw0LjM0NzE3IDEyLjA5Mkw1LjAxMDk0IDEyLjQ2ODNMNS4wMTEyOSA2LjY5NzU0Wk05LjU0NTc5IDUuNzMzNDFMOS41NDU4NCA4LjI5MjA2TDcuMDg4ODYgOS42ODU2NEw2LjQyNTQxIDkuMzA5NDJWNy41MDM0QzYuNzkwMzIgNy4yOTY0OSA5LjU0NTg4IDUuNzI3MTQgOS41NDU3OSA1LjczMzQxWiIgZmlsbD0id2hpdGUiLz4KPC9nPgo8ZGVmcz4KPGNsaXBQYXRoIGlkPSJjbGlwMF80NDFfMTEwMzgiPgo8cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIGZpbGw9IndoaXRlIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==">
      </a>
    <a title="Social" href="https://x.com/pablovelagomez1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
    </a>
  </p>

<p align="center">
  <img src="media/working-sync-optimized.gif" alt="example output" width="720" />
</p>

## Installation
### Using Pixi
Make sure you have the [Pixi](https://pixi.sh/latest/#installation) package manager installed