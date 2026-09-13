/**
 * Builds a Google Form from google_form_manifest.json (pairwise A vs B).
 *
 * Setup:
 * 1. Run: python3 scripts/build_annotation_google_form.py
 * 2. Upload annotation_google_form/google_form_manifest.json to Google Drive.
 * 3. (Recommended) Upload each image to Drive; copy File IDs into manifest paths:
 *    drive_file_id_dalle3, drive_file_id_agentic (or use public_url_* for HTTPS URLs).
 * 4. Paste this file into Extensions → Apps Script bound to a new Google Form,
 *    OR run from standalone script (creates a new form via FormApp.create).
 * 5. Set MANIFEST_FILE_ID below to your uploaded JSON file’s ID (from Drive URL).
 * 6. Run createAnnotationForm() from the Apps Script editor and authorize Drive + Forms.
 *
 * Creates a NEW form each run. Delete old test forms in Drive if needed.
 */

var CONFIG = {
  /** Google Drive file ID of google_form_manifest.json (required). */
  MANIFEST_FILE_ID: '',
  /** Title for the new form. */
  FORM_TITLE: 'Sensory ads — DALL·E 3 vs Agentic editing (pairwise)',
};

/**
 * Entry point: loads manifest from Drive and creates a new Form.
 */
function createAnnotationForm() {
  if (!CONFIG.MANIFEST_FILE_ID) {
    throw new Error('Set CONFIG.MANIFEST_FILE_ID to your manifest JSON file ID on Drive.');
  }
  var jsonText = DriveApp.getFileById(CONFIG.MANIFEST_FILE_ID).getBlob().getDataAsString();
  var manifest = JSON.parse(jsonText);
  return buildFormFromManifest(manifest);
}

/**
 * Optional: paste a small manifest JSON string here for testing without Drive manifest file.
 */
function createAnnotationFormFromHardcodedSample() {
  var sample = {
    layout: 'paired_comparison',
    instructions: 'Sample.',
    comparison_options: [
      'A is clearly better',
      'A is somewhat better',
      'About the same / hard to choose',
      'B is somewhat better',
      'B is clearly better',
    ],
    comparison_dimensions: [
      { id: 'evoke_sensation', prompt: 'Which image better evokes the sensation?' },
      { id: 'convey_action_messages', prompt: 'Which better matches action messages?' },
      { id: 'convey_reason_messages', prompt: 'Which better matches reason messages?' },
      { id: 'persuasive', prompt: 'Which is more persuasive?' },
    ],
    sets: [
      {
        set_index: 1,
        pairs: [
          {
            pair_index_in_set: 1,
            image_key: '0/demo.jpg',
            paths: {
              drive_file_id_dalle3: null,
              drive_file_id_agentic: null,
            },
            sensation: {
              arrow_notation: '0/demo.jpg → touch → 0',
              category: 'touch',
              leaf_index: '0',
              filename: 'demo.jpg',
            },
            messages: {
              action_or_primary: ['Message A1', 'Message A2'],
              reason_or_secondary: ['Message R1'],
            },
          },
        ],
      },
    ],
  };
  return buildFormFromManifest(sample);
}

/**
 * @param {Object} manifest - Same structure as google_form_manifest.json
 * @return {GoogleAppsScript.Form.Form} the created form
 */
function buildFormFromManifest(manifest) {
  var form = FormApp.create(CONFIG.FORM_TITLE);
  form.setDescription(manifest.instructions || '');

  var choices = manifest.comparison_options || [];
  var dims = manifest.comparison_dimensions || [];
  var sets = manifest.sets || [];

  for (var si = 0; si < sets.length; si++) {
    var setObj = sets[si];
    if (si > 0) {
      form.addPageBreakItem();
    }
    form
      .addSectionHeaderItem()
      .setTitle('Set ' + setObj.set_index + ' (10 pairs)')
      .setHelpText('Answer every comparison for each pair in this set. Image A = DALL·E 3, Image B = Agentic editing.');

    var pairs = setObj.pairs || [];
    for (var pi = 0; pi < pairs.length; pi++) {
      addPairBlock(form, setObj.set_index, pairs[pi], dims, choices);
    }
  }

  var url = form.getPublishedUrl();
  Logger.log('Form created: ' + url);
  Logger.log('Edit URL: ' + form.getEditUrl());
  return form;
}

function addPairBlock(form, setIndex, pair, dims, choices) {
  var label = 'S' + setIndex + '-P' + pair.pair_index_in_set + ' — ' + pair.image_key;
  var help = buildPairHelpText(pair);

  form.addSectionHeaderItem().setTitle(label).setHelpText(help);

  var paths = pair.paths || {};
  addPairImages(form, paths);

  for (var di = 0; di < dims.length; di++) {
    var dim = dims[di];
    var qTitle = '[' + label + '] ' + dim.prompt;
    form
      .addMultipleChoiceItem()
      .setTitle(qTitle)
      .setChoiceValues(choices)
      .setRequired(true);
  }
}

function buildPairHelpText(pair) {
  var s = pair.sensation || {};
  var pth = pair.paths || {};
  var lines = [];
  lines.push('Sensation path: ' + (s.arrow_notation || ''));
  lines.push(
    'Category: ' +
      (s.category || '') +
      ' · Variant: ' +
      (s.leaf_index || '') +
      ' · File: ' +
      (s.filename || '')
  );
  lines.push('');
  lines.push('If images do not load, open from ~/experiments/forms/ :');
  lines.push('A (DALL·E 3) — ' + (pth.forms_dalle3_relative || ''));
  lines.push('B (Agentic) — ' + (pth.forms_agentic_relative || ''));

  var msg = pair.messages || {};
  lines.push('');
  lines.push('Action-oriented messages (set 1):');
  appendLines(lines, msg.action_or_primary);
  lines.push('');
  lines.push('Reason-oriented messages (set 2):');
  appendLines(lines, msg.reason_or_secondary);

  return lines.join('\n');
}

function appendLines(lines, arr) {
  if (!arr || !arr.length) {
    lines.push('(none)');
    return;
  }
  for (var i = 0; i < arr.length; i++) {
    lines.push('• ' + arr[i]);
  }
}

function addPairImages(form, paths) {
  paths = paths || {};
  var idA = paths.drive_file_id_dalle3;
  var idB = paths.drive_file_id_agentic;
  var urlA = paths.public_url_dalle3;
  var urlB = paths.public_url_agentic;

  if (idA) {
    try {
      var blobA = DriveApp.getFileById(idA).getBlob();
      form.addImageItem().setTitle('Image A — DALL·E 3').setImage(blobA);
    } catch (e) {
      form.addSectionHeaderItem().setTitle('Image A — Drive load failed (check drive_file_id_dalle3)');
    }
  } else if (urlA) {
    try {
      var blobUrlA = UrlFetchApp.fetch(urlA).getBlob();
      form.addImageItem().setTitle('Image A — DALL·E 3').setImage(blobUrlA);
    } catch (e) {
      form.addSectionHeaderItem().setTitle('Image A — URL fetch failed');
    }
  }

  if (idB) {
    try {
      var blobB = DriveApp.getFileById(idB).getBlob();
      form.addImageItem().setTitle('Image B — Agentic editing').setImage(blobB);
    } catch (e) {
      form.addSectionHeaderItem().setTitle('Image B — Drive load failed (check drive_file_id_agentic)');
    }
  } else if (urlB) {
    try {
      var blobUrlB = UrlFetchApp.fetch(urlB).getBlob();
      form.addImageItem().setTitle('Image B — Agentic editing').setImage(blobUrlB);
    } catch (e) {
      form.addSectionHeaderItem().setTitle('Image B — URL fetch failed');
    }
  }
}
