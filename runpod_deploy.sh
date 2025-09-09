python test_enhanced_supernova.py
✅ Claude API key found, running tests...
🚀 Testing Enhanced SuperNova System
==================================================
🔧 Configuration:
   Claude API Key: ✅ Set
   Remote FinGPT URL: https://gsezd4c8sc7tne-8003.proxy.runpod.net

🌟 Initializing Enhanced SuperNova...
INFO:SuperNovaEnhanced:🌟 SuperNova Enhanced initialized
INFO:SuperNovaEnhanced:🔄 Architecture: Claude + Remote FinGPT + Claude
INFO:httpx:HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 404 Not Found"
WARNING:SuperNovaEnhanced:⚠️ Claude initialization failed: Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model: claude-3-sonnet-20240229'}, 'request_id': 'req_011CSxHRTHYrzhGjEvNC94zk'}
INFO:httpx:HTTP Request: GET https://gsezd4c8sc7tne-8003.proxy.runpod.net/health "HTTP/1.1 200 OK"
INFO:SuperNovaEnhanced:✅ Remote FinGPT server healthy at https://gsezd4c8sc7tne-8003.proxy.runpod.net
INFO:SuperNovaEnhanced:✅ SuperNova Enhanced initialization complete
📊 System Status:
   Architecture: claude_remote_fingpt_claude
   Version: 3.1.0
   claude_preprocessor: ❌ inactive
   remote_fingpt: ✅ active
   local_fingpt_fallback: ❌ inactive
   claude_formatter: ❌ inactive

💬 Testing Enhanced Queries...
------------------------------

🔍 Test 1: I have $10,000 to invest in tech stocks, what do you recommend?
INFO:SuperNovaEnhanced:🧠 Step 1: Claude preprocessing query...
INFO:SuperNovaEnhanced:💰 Step 2: FinGPT financial analysis...
INFO:httpx:HTTP Request: POST https://gsezd4c8sc7tne-8003.proxy.runpod.net/analyze "HTTP/1.1 401 Unauthorized"
WARNING:SuperNovaEnhanced:⚠️ Remote FinGPT failed: Remote FinGPT server error: 401
ERROR:SuperNovaEnhanced:❌ Enhanced analysis failed: No FinGPT instance available (remote or local)
❌ Failed: Analysis failed: No FinGPT instance available (remote or local)
⏳ Waiting 2 seconds...

🔍 Test 2: What's your opinion on Bitcoin for a long-term hold?
INFO:SuperNovaEnhanced:🧠 Step 1: Claude preprocessing query...
INFO:SuperNovaEnhanced:💰 Step 2: FinGPT financial analysis...
ERROR:SuperNovaEnhanced:❌ Enhanced analysis failed: No FinGPT instance available (remote or local)
❌ Failed: Analysis failed: No FinGPT instance available (remote or local)
⏳ Waiting 2 seconds...

🔍 Test 3: I want to diversify my portfolio, suggest 3 different assets
INFO:SuperNovaEnhanced:🧠 Step 1: Claude preprocessing query...
INFO:SuperNovaEnhanced:💰 Step 2: FinGPT financial analysis...
ERROR:SuperNovaEnhanced:❌ Enhanced analysis failed: No FinGPT instance available (remote or local)
❌ Failed: Analysis failed: No FinGPT instance available (remote or local)
⏳ Waiting 2 seconds...

🔍 Test 4: Should I buy Tesla stock right now?
INFO:SuperNovaEnhanced:🧠 Step 1: Claude preprocessing query...
INFO:SuperNovaEnhanced:💰 Step 2: FinGPT financial analysis...
ERROR:SuperNovaEnhanced:❌ Enhanced analysis failed: No FinGPT instance available (remote or local)
❌ Failed: Analysis failed: No FinGPT instance available (remote or local)

🎉 Enhanced SuperNova testing complete!
📈 Conversation history: 0 queries
INFO:SuperNovaEnhanced:🧹 SuperNova Enhanced cleanup complete

==================================================
🌉 Testing SuperNova Bridge Integration
==================================================
🚀 Initializing Enhanced SuperNova (Claude + Remote FinGPT)...
INFO:SuperNovaEnhanced:🌟 SuperNova Enhanced initialized
INFO:SuperNovaEnhanced:🔄 Architecture: Claude + Remote FinGPT + Claude
✨ Enhanced SuperNova initialized (Claude + Remote FinGPT)
🧠 Architecture: Claude → Remote FinGPT → Claude
🚀 Initializing SuperNova Service...
🚀 Initializing Enhanced SuperNova (Claude + Remote FinGPT)...
INFO:SuperNovaEnhanced:🌟 SuperNova Enhanced initialized
INFO:SuperNovaEnhanced:🔄 Architecture: Claude + Remote FinGPT + Claude
✨ Enhanced SuperNova initialized (Claude + Remote FinGPT)
🧠 Architecture: Claude → Remote FinGPT → Claude
📊 Service Status:
   Available: True
   Architecture: claude_remote_fingpt_claude
   Enhanced Enabled: True
   Dual-LLM Enabled: True

💬 Testing Bridge Recommendation...
INFO:httpx:HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 404 Not Found"
WARNING:SuperNovaEnhanced:⚠️ Claude initialization failed: Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model: claude-3-sonnet-20240229'}, 'request_id': 'req_011CSxHS1azqYxpxqENC4bjq'}
INFO:httpx:HTTP Request: GET https://gsezd4c8sc7tne-8003.proxy.runpod.net/health "HTTP/1.1 200 OK"
INFO:SuperNovaEnhanced:✅ Remote FinGPT server healthy at https://gsezd4c8sc7tne-8003.proxy.runpod.net
INFO:SuperNovaEnhanced:✅ SuperNova Enhanced initialization complete
INFO:SuperNovaEnhanced:🧠 Step 1: Claude preprocessing query...
INFO:SuperNovaEnhanced:💰 Step 2: FinGPT financial analysis...
INFO:httpx:HTTP Request: POST https://gsezd4c8sc7tne-8003.proxy.runpod.net/analyze "HTTP/1.1 401 Unauthorized"
WARNING:SuperNovaEnhanced:⚠️ Remote FinGPT failed: Remote FinGPT server error: 401
ERROR:SuperNovaEnhanced:❌ Enhanced analysis failed: No FinGPT instance available (remote or local)
⚠️ Enhanced SuperNova failed: Analysis failed: No FinGPT instance available (remote or local)
🔄 Falling back to Dual-LLM...
❌ Bridge test failed: 'NoneType' object has no attribute 'success'
Traceback (most recent call last):
  File "C:\Users\Nova_\PROJECTS\NovaSignal-Desktop\test_enhanced_supernova.py", line 170, in test_bridge_integration
    if response.success:
       ^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'success'
(.venv) 
