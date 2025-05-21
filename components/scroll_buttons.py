# components/scroll_buttons.py
"""
Component to add fixed scroll-to-top and scroll-to-bottom buttons,
positioned middle-right of the viewport.
"""
import streamlit as st
import uuid # For generating a unique ID for the script instance

UP_ARROW_UNICODE = "\u25B2"  # ▲ BLACK UP-POINTING TRIANGLE
DOWN_ARROW_UNICODE = "\u25BC" # ▼ BLACK DOWN-POINTING TRIANGLE

class ScrollButtons:
    """
    A component to inject HTML, CSS (via style.css), and JavaScript
    for scroll-to-top and scroll-to-bottom buttons, positioned middle-right.
    """

    def __init__(self, visible_threshold_px: int = 200):
        """
        Args:
            visible_threshold_px (int): Scroll distance in pixels after which
                                        the "scroll to top" button becomes visible.
        """
        self.visible_threshold_px = visible_threshold_px
        # Unique ID for the script tag to help with debugging or ensuring script runs once if needed
        self.script_tag_id = f"scroll_buttons_script_instance_{str(uuid.uuid4())[:8]}"

    def _get_button_html(self) -> str:
        """Generates the HTML for the scroll buttons within a container."""
        # IDs are kept simple as they are expected to be unique on the page.
        html_content = f"""
            <div id="scrollButtonContainerFixed" class="scroll-button-container-fixed">
                <div id="scrollTopButtonGlobalV4" class="scroll-button scroll-button-up" title="Scroll to Top">
                    {UP_ARROW_UNICODE}
                </div>
                <div id="scrollBottomButtonGlobalV4" class="scroll-button scroll-button-down" title="Scroll to Bottom">
                    {DOWN_ARROW_UNICODE}
                </div>
            </div>
        """
        return html_content

    def _get_javascript(self) -> str:
        """
        Generates the JavaScript for button functionality.
        This script attempts to be robust against multiple executions by checking
        if listeners have already been attached to the specific elements.
        """
        # The script_tag_id is used to make function names on window unique if needed,
        # and for data attributes to mark elements as having listeners attached by this script instance.
        unique_js_suffix = self.script_tag_id

        javascript_content = f"""
            <script id="{self.script_tag_id}">
                (function() {{
                    // console.log("ScrollButtons script ({self.script_tag_id}) executing.");

                    const scrollTopButton = document.getElementById('scrollTopButtonGlobalV4');
                    const scrollBottomButton = document.getElementById('scrollBottomButtonGlobalV4');
                    const visibleThreshold = {self.visible_threshold_px};

                    // --- Define event handlers ---
                    // These functions are defined fresh each time the script runs.
                    const handleScrollTopClick = () => {{
                        window.scrollTo({{ top: 0, behavior: 'smooth' }});
                    }};

                    const handleScrollBottomClick = () => {{
                        window.scrollTo({{ top: document.body.scrollHeight, behavior: 'smooth' }});
                    }};

                    const checkScrollButtonVisibility = () => {{
                        if (scrollTopButton) {{
                            if (window.pageYOffset > visibleThreshold) {{
                                scrollTopButton.style.display = 'flex';
                            }} else {{
                                scrollTopButton.style.display = 'none';
                            }}
                        }}
                    }};

                    // --- Attach listeners if elements exist and don't already have them from this script ---
                    if (scrollTopButton) {{
                        const listenerFlag = 'data-listener-scrollTop-{unique_js_suffix}';
                        if (!scrollTopButton.getAttribute(listenerFlag)) {{
                            scrollTopButton.addEventListener('click', handleScrollTopClick);
                            scrollTopButton.setAttribute(listenerFlag, 'true');
                            // console.log('Attached click listener to scrollTopButton ({unique_js_suffix})');
                        }}
                    }}

                    if (scrollBottomButton) {{
                        const listenerFlag = 'data-listener-scrollBottom-{unique_js_suffix}';
                        if (!scrollBottomButton.getAttribute(listenerFlag)) {{
                            scrollBottomButton.addEventListener('click', handleScrollBottomClick);
                            scrollBottomButton.setAttribute(listenerFlag, 'true');
                            // console.log('Attached click listener to scrollBottomButton ({unique_js_suffix})');
                        }}
                    }}
                    
                    // --- Manage window scroll listener ---
                    // Store the actual function reference on the window object with a unique key.
                    // This allows us to remove the exact same listener instance if this script block re-runs.
                    const windowScrollListenerKey = '_scrollButtonWindowListener_{unique_js_suffix}';

                    // If a listener from a previous run of THIS SCRIPT INSTANCE exists, remove it.
                    if (window[windowScrollListenerKey] && typeof window[windowScrollListenerKey] === 'function') {{
                        window.removeEventListener('scroll', window[windowScrollListenerKey]);
                        // console.log('Removed old window scroll listener for {unique_js_suffix}');
                    }}
                    
                    // Store the current visibility check function and add it as the new listener.
                    window[windowScrollListenerKey] = checkScrollButtonVisibility;
                    window.addEventListener('scroll', window[windowScrollListenerKey]);
                    // console.log('Added new window scroll listener for {unique_js_suffix}');

                    // Initial visibility check for the scroll-to-top button
                    checkScrollButtonVisibility();

                }})(); // IIFE
            </script>
        """
        return javascript_content

    def render(self) -> None:
        """
        Renders the scroll buttons by injecting HTML and JavaScript.
        """
        button_html = self._get_button_html()
        button_js = self._get_javascript()
        
        st.markdown(button_html, unsafe_allow_html=True)
        # Using st.components.v1.html is generally robust for script injection.
        # The height=0 div is a common pattern to inject JS without visible HTML output from this call.
        st.components.v1.html(f"<div style='height:0; visibility:hidden;'>{button_js}</div>", height=0)
        # print(f"ScrollButtons component ({self.script_tag_id}) rendered.") # For server-side logging
