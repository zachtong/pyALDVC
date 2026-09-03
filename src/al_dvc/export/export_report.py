"""PDF report of a run (matplotlib PdfPages)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from ..core.config import para_to_dict  # noqa: E402
from ..core.data_structures import STATUS_NAMES, PipelineResult  # noqa: E402
from ..viz.slices import histogram_panel, plot_field_slices  # noqa: E402
from .export_utils import ensure_dir, field_array, result_summary  # noqa: E402


def _text_page(pdf: PdfPages, title: str, lines: list[str], width: int = 100) -> None:
    import textwrap

    wrapped: list[str] = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(line)
        else:
            indent = len(line) - len(line.lstrip())
            wrapped.extend(textwrap.wrap(line, width=width, subsequent_indent=" " * (indent + 4)))
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.96, title, fontsize=16, weight="bold", va="top")
    y = 0.92
    for line in wrapped:
        fig.text(0.05, y, line, fontsize=8.5, family="monospace", va="top")
        y -= 0.016
        if y < 0.04:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            y = 0.96
    pdf.savefig(fig)
    plt.close(fig)


def export_report(
    result: PipelineResult, path: str | Path, gt: dict | None = None, fields: list[str] | None = None, max_frames: int = 6
) -> Path:
    """Write a multi-page PDF: parameters, timings, convergence, field slices.

    ``gt`` may contain ``"U"`` (list of (N,3) ground-truth displacements per
    frame) and/or ``"F"`` (list of (N,3,3)) to add error statistics.
    """
    p = Path(path)
    if p.suffix.lower() != ".pdf":
        p = p.with_suffix(".pdf")
    ensure_dir(p.parent)
    mesh = result.dvc_mesh
    para = result.dvc_para
    summary = result_summary(result)
    units = para.units
    if fields is None:
        fields = ["disp_u", "disp_v", "disp_w"]
        if result.result_disp and result.result_disp[0].U_std is not None:
            fields.append("disp_std")
        if result.result_strain:
            fields += ["exx", "eyy", "ezz", "exy", "exz", "eyz", "von_mises"]

    with PdfPages(str(p)) as pdf:
        lines = [
            f"volume (z,y,x): {result.volume_shape}   node grid (z,y,x): {mesh.grid_shape}   nodes: {mesh.n_nodes}",
            f"spacing (x,y,z): {mesh.spacing}   valid nodes: {int(mesh.node_valid.sum())}",
            "",
        ]
        lines.append("Parameters:")
        for k, v in para_to_dict(para).items():
            lines.append(f"  {k:<28} {v}")
        lines.append("")
        lines.append("Timings [s]:")
        for k, v in result.timings.items():
            lines.append(f"  {k:<28} {v:9.2f}")
        lines.append("")
        lines.append("Frames:")
        for fr in summary["frames"]:
            lines.append("  " + ", ".join(f"{k}={v}" for k, v in fr.items() if k != "update_global"))
        if result.stopped_early:
            lines.append("")
            lines.append(f"STOPPED EARLY at frame {result.stopped_at_frame}: {result.stop_reason}")
        _text_page(pdf, "pyALDVC run report", lines)

        # convergence / quality page per frame
        for k, fr in enumerate(result.result_disp[:max_frames]):
            fig, axes = plt.subplots(2, 3, figsize=(13, 8))
            axes = axes.ravel()
            if fr.zncc is not None:
                histogram_panel(axes[0], fr.zncc, f"frame {k + 1}: final ZNCC")
            if fr.status is not None:
                st = np.asarray(fr.status)
                codes, counts = np.unique(st, return_counts=True)
                axes[1].bar([STATUS_NAMES.get(int(c), str(c)) for c in codes], counts, color="#dd8452")
                axes[1].set_title("node status")
                axes[1].tick_params(axis="x", rotation=30)
            if fr.admm is not None:
                a = fr.admm
                steps = np.arange(2, 2 + len(a.update_global))
                axes[2].semilogy(steps, np.maximum(a.update_global, 1e-16), "o-", label="|dU| global")
                axes[2].semilogy(steps, np.maximum(a.update_local, 1e-16), "s-", label="|dU| local")
                axes[2].axhline(para.admm_tol, color="k", ls="--", lw=1, label="admm_tol")
                axes[2].set_title(f"ADMM updates (beta={a.beta:.2e}, mu={a.mu:.1e})")
                axes[2].set_xlabel("ADMM step")
                axes[2].legend(fontsize=8)
                axes[3].semilogy(
                    np.arange(1, 1 + len(a.primal_residual_u)),
                    np.maximum(a.primal_residual_u, 1e-16),
                    "o-",
                    label="rms |u - u_hat|",
                )
                axes[3].semilogy(
                    np.arange(1, 1 + len(a.primal_residual_f)),
                    np.maximum(a.primal_residual_f, 1e-16),
                    "s-",
                    label="rms |F - grad u_hat|",
                )
                axes[3].set_title("primal residuals")
                axes[3].legend(fontsize=8)
                if a.beta_sweep is not None:
                    sw = a.beta_sweep
                    ax4 = axes[4]
                    ax4.semilogx(sw["betas"], sw["score"], "o-", color="#4c72b0", label="L-curve score")
                    ax4.axvline(a.beta, color="k", ls="--", lw=1, label=f"beta = {a.beta:.2e}")
                    ax4.set_xlabel("beta")
                    ax4.set_ylabel("normalised |u-u_hat| + |F-grad u_hat|")
                    ax4.set_title("beta auto-tuning (L-curve sweep)")
                    ax4.legend(fontsize=8)
                    ax4b = ax4.twinx()
                    ax4b.semilogx(sw["betas"], sw["err1"], "s:", color="#dd8452", ms=4, label="|u-u_hat|")
                    ax4b.semilogx(
                        sw["betas"],
                        sw["err2"] * float(np.mean(para.winstepsize)) ** 2,
                        "^:",
                        color="#55a868",
                        ms=4,
                        label="|F-grad u_hat| x h^2",
                    )
                    ax4b.legend(fontsize=7, loc="upper center")
                else:
                    iters = [li.n_iter[li.status == 0] for li in a.local_info]
                    axes[4].boxplot([it if it.size else [0] for it in iters])
                    axes[4].set_title("IC-GN iterations per ADMM step")
            else:
                axes[2].axis("off")
                axes[3].axis("off")
                axes[4].axis("off")
            if fr.U_local is not None:
                d = np.linalg.norm(fr.U - fr.U_local, axis=1)
                histogram_panel(axes[5], d, "|U_final - U_local| [voxel]")
            else:
                axes[5].axis("off")
            fig.suptitle(f"Frame {k + 1} (reference {fr.ref_frame}) diagnostics")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # field pages
        for k in range(min(len(result.result_disp), max_frames)):
            for name in fields:
                try:
                    vals = field_array(result, k, name)
                except ValueError:
                    continue
                fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
                label = f"{name} ({units})" if name.startswith("disp") else name
                plot_field_slices(mesh.to_grid(vals), mesh, title=f"frame {k + 1} {label}", fig=fig, axes=axes)
                pdf.savefig(fig)
                plt.close(fig)

        # ground-truth error pages
        if gt is not None:
            for k, fr in enumerate(result.result_disp[:max_frames]):
                U_gt = gt.get("U", [None] * (k + 1))[k] if k < len(gt.get("U", [])) else None
                if U_gt is None:
                    continue
                U = fr.U_accum if fr.U_accum is not None else fr.U
                err = U - U_gt
                ok = mesh.node_valid & np.all(np.isfinite(err), axis=1)
                fig, axes = plt.subplots(2, 3, figsize=(13, 8))
                for c, nm in enumerate("uvw"):
                    histogram_panel(axes[0, c], err[ok, c], f"frame {k + 1}: error {nm} [voxel]", gt=0.0)
                    plot_field_slices(
                        mesh.to_grid(np.where(ok, err[:, c], np.nan)),
                        mesh,
                        title=f"err {nm}",
                        fig=fig,
                        axes=[axes[1, c]] * 3,
                        cmap="coolwarm",
                    )
                rmse = np.sqrt(np.mean(err[ok] ** 2, axis=0))
                fig.suptitle(
                    f"Frame {k + 1} displacement error vs ground truth: "
                    f"RMSE u,v,w = {rmse[0]:.4f}, {rmse[1]:.4f}, {rmse[2]:.4f} voxel"
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    return p
