from types import SimpleNamespace
import qualify
import planner
from schemas import SearchRequest, EvidencePlan

def test_visual_review_is_separate_from_website_text(monkeypatch):
    calls=[]
    def call(prompt,schema,images=None):
        calls.append((prompt,schema,images))
        if schema is qualify.VisualReview:
            return qualify.VisualReview(observations=["Desktop navigation overlaps"], summary="Navigation overlaps")
        return "analysis"
    monkeypatch.setattr(qualify,"structured_call",call)
    site={"text":"Gym in Lahore", "footprint":"measured", "desktop_screenshot":"image1", "mobile_screenshot":"image2"}
    state={"service":"Web Development","industry":"gyms","location":"Lahore", "plan":{"ideal_buyer":"Gyms in Lahore","need_indicators":[],"disqualifiers":[],"visual_indicators":["Broken navigation"]}}
    assert qualify.judge_company({"title":"Gym", "url":"https://example.com", "snippet":"Gym"},site,state)=="analysis"
    assert calls[0][2]==["image1","image2"]
    assert "Gym in Lahore" not in calls[0][0]
    assert calls[1][2] is None
    assert "Desktop navigation overlaps" in calls[1][0]

def test_planner_distinguishes_seller_service_from_buyer_business(monkeypatch):
    prompts=[]
    monkeypatch.setattr(planner,"structured_call",lambda prompt,schema: prompts.append(prompt) or EvidencePlan(ideal_buyer="Gyms"))
    planner._generate_research_plan(SearchRequest(location="Lahore",industry="gyms",service="Web Development",number_of_leads=10))
    assert "Never require gyms to offer web development" in prompts[0]
    assert "at least 3 must describe a gap" not in prompts[0]
