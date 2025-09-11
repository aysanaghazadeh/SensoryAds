SENSATION_HIERARCHY = {
    'Touch': {
        'Temperature': [
            'Freezing Cold',
            'Cool and Refreshing',
            'Comforting Warmth',
            'Intense Heat'
        ],
        'Texture': [
            'Softness',
            'Silky Smoothness',
            'Stickiness',
            'Roughness',
            'Sharpness'
        ],
        'Moisture and Dryness': [
            'Soaking Wetness',
            'Mistiness',
            'Greasiness and Oiliness',
            'Dryness'
        ],
        'Motion and Weight': [
            'High Speed and Acceleration',
            'Weightlessness',
            'Heaviness',
            'Tension',
            'Vibration'
        ],
        'Pain and Relief': [
            'Sharp Pain',
            'Aching Pain',
            'Soothing Relief and Numbing'
        ]
    },
    'Smell': {
        'Fresh and Clean Smell': [
            'Fresh Fruit Smell',
            'Fresh Greenery Smell',
            'Clean and Soapy Smell',
            'Refreshing Minty Smell',
            'Refreshing Ocean Smell',
            'Floral Smell',
            'Fragrant Smell'
        ],
        'Culinary Smell': [
            'Drinks Smell',
            'Savory Food Smell',
            'Spices Smell',
            'Bakery Smell',
            'Sweet Smell'
        ],
        'Earthy and Musky Smell': [
            'Woody Smell',
            'Leather Smell',
            'Earth and Soil Smell',
            'Natural Greenery Smell'
        ],
        'Chemical and Pungent Smell':[
            'Medicine  and Antiseptic Smell',
            'Cleaning Products and Chemicals Smells',
            'Pungent Smell',
            'Sickening Smell'
        ]
    },
    'Taste': [
        'Sweet Taste',
        'Sour Taste',
        'Bitter Taste',
        'Spicy Taste',
        'Umami Taste',
        'Salty Taste',
        'Cooling Minty Taste'
    ],
    'Sound': {
        'Music Sounds': [
            'Instruments Sound',
            'Singing Voice'
        ],
        'Rural and Industrial Sounds': [
            'Vehicles Sound',
            'Traffic Jam Sound',
            'Machinery Sound'
        ],
        'Nature Sounds': [
            'Animal Sounds',
            'Natural Water Sound',
            'Atmospheric Phenomena Sound',
            'Geological Events Sound',
            'Peaceful Ambient'
        ],
        'Liquid Sounds': [
            'Bubbling and Fizzing Sound',
            'Liquid Splash Sound',
            'Liquid Pouring Sound'
        ],
        'Silence and Quiet': None,
        'Human Voice': [
            'Sad Voice',
            'Rushing Voice',
            'Compassion Voice',
            'Happy Voice',
            'Argument Voice',
            'Harsh Voice'
        ]
    },
    'Sight': [
        'Brilliance and Glow',
        'Clarity',
        'Blur',
        'Darkness'
    ]
    ,
    'None': None
}

SENSATION_DEFINITION = {
    'root': 
        """
        <li>Touch: sensation received by the skin. Ex. Temperature Sensation (Warm, Cold, etc), Motion Sensation (Speed, etc.)</li>
        <li>Smell: sensation received by the nose. Ex. Floral Smell, Bakery Smell</li>
        <li>Taste: sensation received by the tongue. Ex. Sour Taste, Bitter Taste</li>
        <li>Sight: sensation received by the eyes. Ex. Darkness, Brilliance</li>
        <li>Sound: sensation received by the ears. Ex. Music Sound, Nature Sound </li>
        <li>None: The image does not evoke any sensation.</li>
        """,
    'Touch':
        """
        <li>Temerature: perceptions of heat or cold detected by thermoreceptors in the skin.</li>
        <li>Texture: perceptions of a surface like smoothness, roughness, or structure detected by mechanoreceptors.</li>
        <li>Moisture and Dryness: awareness of the presence or absence of water, oil, or other liquids.</li>
        <li>Motion and Weight: awareness of body motion, acceleration, relative position, and weight from proprioceptive and vestibular input.</li>
        <li>Pain and Relief Sensation: unpleasant or soothing feelings from nociceptors or comfort-inducing stimuli.</li>
        """,
    'Smell':
        """
        <li> Fresh and Clean Smell: scents associated with purity, nature, and cleanliness.</li>
		<li> Culinary Smell: aromas linked to prepared foods, drinks, and baked goods.</li>
		<li> Earthy and Musky Smell: deep, natural aromas resembling soil, wood, greenery, that reminds of nature.</li>
		<li> Chemical and Pungent: strong, sharp odors associated with chemicals, medicine, or disinfectants.</li>
        """,
    'Taste': 
        """
        <li>Sweet Taste: taste of sugars, often pleasant and dessert-like.</li>
		<li>Sour Taste: acidic taste that causes mouth puckering.</li>
		<li>Bitter Taste: sharp, often unpleasant taste from alkaloids.</li>
		<li>Salty Taste: taste from sodium or mineral salts.</li>
		<li>Umami Taste: savory taste from glutamates (meaty, brothy).</li>
		<li>Spicy Taste: burning heat sensation from chili compounds.</li>
		<li>Cooling Minty Taste: cooling sensation from mint or menthol.</li>
        """,
    'Sound':
        """
        <li>Music Sounds: organized sounds from instruments or human voice.</li>
        <li>Rural and Industrial Sounds: are human-influenced environmental soundscapes spanning countryside and manufacturing settings, encompassing vehicles and traffic-jam noise as well as machinery from farms, factories, and construction sites.</li>
        <li>Nature Sounds: Any sound that can be heard in nature.</li>
        <li>Liquid Sounds: noises created by moving or interacting liquids.</li>
        <li>Silence and Quiet: absence or near absence of sound.</li>
        <li>Human Voice: Different sounds heard from someone talking.</li>
        """,
    'Sight':
        """
        <li>Brilliance and Glow: intense brightness or vividness or steady radiance from a surface or object.</li>
		<li>Clarity: sharp, well-defined visual detail.</li>
		<li>Blur: lack of sharpness or focus.</li>
        """,
    'Temperature':
        """
        <li>Freezing Cold: sensation of extreme cold, often causing numbness or discomfort.</li>
		<li>Cool and Refreshing: mildly cold feeling that is pleasant or invigorating.</li>
		<li>Comforting Warmth: gentle heat that feels soothing and pleasant.</li>
		<li>Intense Heat: high temperature sensation that may cause discomfort or pain.</li>
        """,
    'Texture':
        """
        <li>Softness: smooth and yielding surface to the touch.</li>
        <li>Silky Smoothness: extremely smooth, flowing texture like silk.</li>
        <li>Stickiness: surface that resists motion due to adhesion.</li>
        <li>Roughness: uneven surface with coarse points or bumps.</li>
        <li>Sharpness: pointed or edged surface that can cut or prick.</li>
        """,
    'Moisture and Dryness':
        """
        <li>Soaking Wetness: sensation of being fully saturated with liquid.</li>
		<li>Mistiness: light, fine moisture on the skin.</li>
		<li>Greasiness and Oiliness: slick, oily coating on the surface.</li>
		<li>Dryness: absence of moisture, often rough or tight-feeling.</li>
        """,
    'Motion and Weight':
        """
        <li>High Speed and Acceleration: sensation of rapid movement or change in velocity.</li>
		<li>Weightlessness: feeling of no gravity or reduced body weight.</li>
		<li>Heaviness: increased load or downward pressure on the body.</li>
		<li>Tension: tightness in muscles or connective tissue.</li>
		<li>Vibration: rapid, repeated movement or oscillation against the skin.</li>
        """,
    'Pain and Relief':
        """
        <li>Sharp Pain: sudden, intense pain from a pointed source.</li>
		<li>SAching Pain: deep, dull, continuous discomfort.</li>
		<li>Soothing Relief and Numbing: reduction or dulling of discomfort.</li>    
        """,
    'Fresh and Clean Smell':
        """
        <li>Fresh Fruit Smell: crisp, plant-based aroma from fruits including citric and other fruits.</li>
		<li>Fresh Greenery Smell: clean, leafy, outdoor scent.</li>
		<li>Clean and Soapy Smell: scent of cleanliness, often associated with detergents or soap.</li>
		<li>Refreshing Minty Smell: Cooling scent of mint.</li>
		<li>Refreshing Ocean Smell: Smell of sea, ocean, etc. reminding of refreshness of ocean.</li>
		<li>Floral Smell: Smell of flowers.</li>
		<li>Fragrant Smell: Smell of perfume, cologne, etc.</li>
        """,
    'Culinary Smell':
        """
        <li>Drinks Smell: aromas of beverages like coffee, wine, or juice.</li>
		<li>Savory Foods Smell: scent of cooked or prepared meals.</li>
		<li>Spices Smell: Smells like pepper, tumeric, etc. </li>
		<li>Bakery Smell: aroma of fresh bread, and baked goods.</li>
		<li> Sweet Smell: Smells like chocolate, vanilla, etc. which remind you of sweet taste. </li>
        """,
    'Earthy and Musky Smell':
        """
        <li>Woody Smell: scent of fresh-cut wood or forest.</li>
		<li>Leather Smell: rich, slightly musky smell of tanned hide.</li>
		<li>Earth and Soil Smell: damp, mineral-rich scent of soil.</li>
		<li>Natural Greenery Smell: smell of plants, jungles, farms, and other greenery reminding you of the natural green places.</li>
        """,
    'Chemical and Pungent Smell':
        """
        <li>Medicine and Antiseptic Smell: sharp, clinical odor like alcohol</li>
		<li>Cleaning Products and Chemical Smells: strong, synthetic scent from cleansers like bleech.</li>
		<li>Pungent Smell: A sharp and bad smell like cigar, sweat, onion, garlic, etc.</li>
		<li>Sickening Smell: Advance pungent smells like rotting mean, sewage, sulphuric acid, etc.</li>
        """,
    'Music Sounds':
        """
        <li>Instruments Sound: sound produced by musical tools like guitar, piano, drums.</li>
		<li>Singing Voice: human vocal musical sound.</li>
        """,
    'Rural and Industrial Sounds':
        """
        <li>Vehicles Sound: Sound of cars, trains, airplanes, etc.</li>
		<li>Traffic Jam Sound: Sound of horn, cars, etc.</li>
		<li>Machinary Sound: Sound of machines in factories and industrial equipment, construction, etc.</li>
        """,
    'Nature Sounds':
        """
        <li>Animal Sounds: Sound made by animals in nature.</li>
		<li>Natural Water Sounds: Sound of waterfall, ocean, sea, river, etc.</li>
		<li>Atmospheric Phenomena: Sound made by different weather conditions.</li>
		<li>Geological Events: Sound of volcano, earthquake, flood</li>
		<li>Peaceful Ambient: soft, calm natural background noise.</li>
        """,
    'Liquid Sounds':
        """
        <li>Bubbling and Fizzing Sound: rapid release of air or gas in liquid.</li>
		<li>Splash Sound: sound of something hitting liquid.</li>
		<li>Pouring Sound: continuous liquid flow sound.</li>
        """,
    'Silence and Quiet': None,
    'Human Voice':
        """
        <li>Sad Voice: Tone of a person who is sad.</li>
		<li>Rushing Voice: Tone of a person in hurry.</li>
		<li>Compassion Voice: Tone of a person trying to comfort someone.</li>
		<li>Happy Voice: Tone of a person who is happy.</li>
		<li>Argument Voice: People arguing over a topic, ex: Political Debate </li>
		<li>Harsh Voice: Someone shouting</li>
        """,
    'None': None
}

SENSATIONS_PARENT_MAP = {'Aching Pain': 'Pain and Relief',
                         'Animal Sounds': 'Nature Sounds',
                         'Argument Voice': 'Human Voice',
                         'Atmospheric Phenomena Sound': 'Nature Sounds',
                         'Bakery Smell': 'Culinary Smell',
                         'Bitter Taste': 'Taste',
                         'Blur': 'Sight',
                         'Brilliance and Glow': 'Sight',
                         'Bubbling and Fizzing Sound': 'Liquid Sounds',
                         'Chemical and Pungent Smell': 'Smell',
                         'Clarity': 'Sight',
                         'Clean and Soapy Smell': 'Fresh and Clean Smell',
                         'Cleaning Products and Chemicals Smells': 'Chemical and Pungent Smell',
                         'Comforting Warmth': 'Temperature',
                         'Compassion Voice': 'Human Voice',
                         'Cool and Refreshing': 'Temperature',
                         'Cooling Minty Taste': 'Taste',
                         'Culinary Smell': 'Smell',
                         'Darkness': 'Sight',
                         'Drinks Smell': 'Culinary Smell',
                         'Dryness': 'Moisture and Dryness',
                         'Earth and Soil Smell': 'Earthy and Musky Smell',
                         'Earthy and Musky Smell': 'Smell',
                         'Floral Smell': 'Fresh and Clean Smell',
                         'Fragrant Smell': 'Fresh and Clean Smell',
                         'Freezing Cold': 'Temperature',
                         'Fresh Fruit Smell': 'Fresh and Clean Smell',
                         'Fresh Greenery Smell': 'Fresh and Clean Smell',
                         'Fresh and Clean Smell': 'Smell',
                         'Geological Events Sound': 'Nature Sounds',
                         'Greasiness and Oiliness': 'Moisture and Dryness',
                         'Happy Voice': 'Human Voice',
                         'Harsh Voice': 'Human Voice',
                         'Heaviness': 'Motion and Weight',
                         'High Speed and Acceleration': 'Motion and Weight',
                         'Human Voice': 'Sound',
                         'Instruments Sound': 'Music Sounds',
                         'Intense Heat': 'Temperature',
                         'Leather Smell': 'Earthy and Musky Smell',
                         'Liquid Pouring Sound': 'Liquid Sounds',
                         'Liquid Sounds': 'Sound',
                         'Liquid Splash Sound': 'Liquid Sounds',
                         'Machinery Sound': 'Rural and Industrial Sounds',
                         'Medicine  and Antiseptic Smell': 'Chemical and Pungent Smell',
                         'Mistiness': 'Moisture and Dryness',
                         'Moisture and Dryness': 'Touch',
                         'Motion and Weight': 'Touch',
                         'Music Sounds': 'Sound',
                         'Natural Greenery Smell': 'Earthy and Musky Smell',
                         'Natural Water Sound': 'Nature Sounds',
                         'Nature Sounds': 'Sound',
                         'None': 'None',
                         'Pain and Relief': 'Touch',
                         'Peaceful Ambient': 'Nature Sounds',
                         'Pungent Smell': 'Chemical and Pungent Smell',
                         'Refreshing Minty Smell': 'Fresh and Clean Smell',
                         'Refreshing Ocean Smell': 'Fresh and Clean Smell',
                         'Roughness': 'Texture',
                         'Rural and Industrial Sounds': 'Sound',
                         'Rushing Voice': 'Human Voice',
                         'Sad Voice': 'Human Voice',
                         'Salty Taste': 'Taste',
                         'Savory Food Smell': 'Culinary Smell',
                         'Sharp Pain': 'Pain and Relief',
                         'Sharpness': 'Texture',
                         'Sickening Smell': 'Chemical and Pungent Smell',
                         'Sight': 'None',
                         'Silence and Quiet': 'Sound',
                         'Silky Smoothness': 'Texture',
                         'Singing Voice': 'Music Sounds',
                         'Smell': 'None',
                         'Soaking Wetness': 'Moisture and Dryness',
                         'Softness': 'Texture',
                         'Soothing Relief and Numbing': 'Pain and Relief',
                         'Sound': 'None',
                         'Sour Taste': 'Taste',
                         'Spices Smell': 'Culinary Smell',
                         'Spicy Taste': 'Taste',
                         'Stickiness': 'Texture',
                         'Sweet Smell': 'Culinary Smell',
                         'Sweet Taste': 'Taste',
                         'Taste': 'None',
                         'Temperature': 'Touch',
                         'Tension': 'Motion and Weight',
                         'Texture': 'Touch',
                         'Touch': 'None',
                         'Traffic Jam Sound': 'Rural and Industrial Sounds',
                         'Umami Taste': 'Taste',
                         'Vehicles Sound': 'Rural and Industrial Sounds',
                         'Vibration': 'Motion and Weight',
                         'Weightlessness': 'Motion and Weight',
                         'Woody Smell': 'Earthy and Musky Smell'}
