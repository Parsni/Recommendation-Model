"""Capture live UI screenshots from the two running Streamlit apps.

Uses the locally installed Microsoft Edge through Playwright (no bundled browser
download needed). Start the servers first:

    streamlit run app_router.py --server.port 8501
    streamlit run app_prompt.py --server.port 8502

Output: outputs/screenshots/live_2026/*.png
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "outputs" / "screenshots" / "live_2026"
ROUTER = "http://localhost:8501"
PROMPT = "http://localhost:8502"
PROMPT_TEXT = ("Draft a short apology email to a client whose delivery slipped by "
               "three days, and offer a revised timeline.")


def settle(page, ms=1800):
    page.wait_for_timeout(ms)


def shot(page, name, element_text=None, full=False):
    path = OUT / f"{name}.png"
    if element_text:
        page.locator("details").filter(has_text=element_text).first.screenshot(path=str(path))
    else:
        page.screenshot(path=str(path), full_page=full)
    print("saved", path.name)


def click(page, name, role="button"):
    try:
        page.get_by_role(role, name=name, exact=False).first.click(timeout=8000)
        settle(page)
        return True
    except Exception:
        print(f"  {role} not found: {name!r}")
        return False


def pick(page, name):
    try:
        page.locator("[data-testid=stRadio] label").filter(has_text=name).first.click(timeout=8000)
        page.wait_for_timeout(4500)
        return True
    except Exception:
        print(f"  radio option not found: {name!r}")
        return False


def open_expander(page, name, wait=2600):
    try:
        page.locator("details").filter(has_text=name).first.locator("summary").click(timeout=8000)
        page.wait_for_timeout(wait)
        return True
    except Exception:
        print(f"  expander not found: {name!r}")
        return False


def scroll_to(page, text, offset=-90):
    """Bring a heading/caption starting with `text` to the top of the app region."""
    import re
    pat = re.compile(re.escape(text))
    hits = page.locator("h1, h2, h3, h4, h5, h6, p, summary, label").filter(has_text=pat)
    if hits.count() == 0:
        print(f"  scroll target missing: {text!r}")
        return False
    target = hits.first
    try:
        target.scroll_into_view_if_needed(timeout=6000)
    except Exception as e:
        print(f"  scroll failed for {text!r}: {type(e).__name__}")
        return False
    page.evaluate(
        """(off) => {
            const cands = [...document.querySelectorAll('*')].filter(
                e => e.scrollHeight > e.clientHeight + 200 &&
                     ['auto','scroll'].includes(getComputedStyle(e).overflowY));
            const sc = cands.sort((a,b)=>b.scrollHeight-a.scrollHeight)[0];
            if (sc) sc.scrollTop += off;
        }""",
        offset,
    )
    page.wait_for_timeout(1500)
    return True


def scroll_top(page):
    page.evaluate("const s=document.querySelector('section.stMain');"
                  "if (s) s.scrollTo(0,0); window.scrollTo(0,0);")
    page.wait_for_timeout(1000)


def capture_router(page):
    page.goto(ROUTER, wait_until="networkidle")
    settle(page, 3500)
    shot(page, "r1_step1_prompt_and_templates")

    click(page, "Executive Email")
    settle(page, 1500)
    shot(page, "r2_step2_optimization_priority")

    click(page, "Launch Analysis Pipeline")
    settle(page, 7000)
    scroll_to(page, "In-Flight Structural Featurization", -60)
    shot(page, "r3_decision_trail_six_stages")
    scroll_to(page, "The Algorithm Selected", 60)
    shot(page, "r4_winner_reveal")

    if open_expander(page, "Inspect Complete 6-Stage"):
        shot(page, "r5_stage_mathematics", element_text="Inspect Complete 6-Stage")

    scroll_to(page, "Routing Accuracy", 130)
    shot(page, "r6_accuracy_dual_metrics")

    scroll_to(page, "Decision Analytics", 130)
    shot(page, "r7_decision_analytics")

    scroll_to(page, "Under the Hood", 130)
    shot(page, "r8_under_the_hood")

    if open_expander(page, "Failsafe gate, audit trail"):
        shot(page, "r9_audit_trail", element_text="Failsafe gate, audit trail")

    scroll_to(page, "Candidate Model Comparison Deck", 130)
    shot(page, "r10_comparison_deck")

    scroll_to(page, "Ask the Router", 130)
    try:
        page.locator('textarea[aria-label^="Ask"], [data-testid="stChatInput"] textarea').first.fill(
            "Why was this model chosen over the runner-up?")
        page.keyboard.press("Enter")
        settle(page, 5000)
    except Exception as e:
        print("  chat input failed:", e)
    shot(page, "r11_chat_assistant")

    click(page, "Continue to Feedback")
    settle(page, 5000)
    shot(page, "r12_feedback_page", full=True)
    page.goto(ROUTER, wait_until="networkidle")
    settle(page, 4000)

    pick(page, "Simple")
    settle(page, 4500)
    scroll_top(page)
    shot(page, "r13_simple_mode")

    pick(page, "Expert")
    settle(page, 4500)
    shot(page, "r14_expert_mode", full=True)


def click_tab(page, name):
    try:
        page.get_by_role("tab", name=name, exact=False).first.click(timeout=8000)
        page.wait_for_timeout(4000)
        return True
    except Exception:
        print(f"  tab not found: {name!r}")
        return False


def capture_prompt(page):
    page.goto(PROMPT, wait_until="networkidle")
    settle(page, 4500)
    shot(page, "p1_find_a_model_home")

    page.locator("textarea, input[type=text]").first.fill(PROMPT_TEXT)
    click(page, "Find my model")
    settle(page, 7000)
    shot(page, "p2_best_match_grid")
    shot(page, "p2b_best_match_full", full=True)

    open_expander(page, "How these rankings work")
    scroll_to(page, "How these rankings work", -150)
    shot(page, "p3_ranking_methodology")

    click(page, "Browse")
    settle(page, 5000)
    shot(page, "p4_carousel")

    if pick(page, "My trial log"):
        shot(page, "p5_trial_log", full=True)
        if click_tab(page, "Log a trial"):
            scroll_top(page)
            shot(page, "p6_log_a_trial")

    if pick(page, "Report & checks"):
        click(page, "Run evaluation")
        settle(page, 10000)
        shot(page, "p7_report_model_comparison", full=True)
        if click_tab(page, "Data quality"):
            shot(page, "p8_data_quality", full=True)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000},
                                  device_scale_factor=1.25)
        page = ctx.new_page()
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        for fn in (capture_router, capture_prompt):
            try:
                fn(page)
            except Exception as e:
                print(f"{fn.__name__} aborted:", type(e).__name__, e)
        print("console errors:", len(errors))
        for e in errors[:10]:
            print("  !", e[:200])
        browser.close()


if __name__ == "__main__":
    sys.exit(run())
