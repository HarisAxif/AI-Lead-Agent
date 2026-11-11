"""Shared service definitions used across request validation and research."""

MENU_SERVICE_RULES = """
SERVICE TERMINOLOGY FOR THIS AGENT:
- "Virtual menu" means an augmented reality (AR) menu: customers preview
  dishes as interactive 3D models, including placing them in their real
  surroundings through a phone camera. Preserve this meaning when extracting,
  clarifying, planning, and qualifying the service.
- A digital menu means an ordinary web, QR-code, PDF, or online ordering menu.
  Do not reinterpret an explicit digital menu request as an AR menu.
- A QR code alone, dish photos, a PDF, online ordering, or a responsive menu
  does not establish that an AR menu exists or that the AR opportunity is solved.
- For virtual/AR menus, assess observable fit for 3D dish previews (for example
  a visually distinctive food offering or an experiential dine-in concept).
  Missing prices, a missing menu button, or poor mobile layout alone do not
  establish need for AR. Do not substitute ordinary digital-menu improvements.
- Existing digital menus do not disqualify an AR prospect. Only verified
  equivalent AR/3D dish-preview functionality can establish that capability.
- Absence of AR wording on inspected pages does not prove absence of AR across
  the business. If only buyer fit is observable, label the plan fit-only rather
  than claiming verified AR need, adoption intent, budget, or willingness to buy.
"""
