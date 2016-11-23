<!DOCTYPE HTML>
<html>
<head>
<title>References</title>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.2/jquery.min.js" type="text/javascript"></script>
<script type="text/javascript">
<!--
// QuickSearch script for JabRef HTML export 
// Version: 3.0
//
// Copyright (c) 2006-2011, Mark Schenk
//
// This software is distributed under a Creative Commons Attribution 3.0 License
// http://creativecommons.org/licenses/by/3.0/
//
// Features:
// - intuitive find-as-you-type searching
//    ~ case insensitive
//    ~ ignore diacritics (optional)
//
// - search with/without Regular Expressions
// - match BibTeX key
//

// Search settings
var searchAbstract = true;	// search in abstract
var searchReview = true;	// search in review

var noSquiggles = true; 	// ignore diacritics when searching
var searchRegExp = false; 	// enable RegExp searches


if (window.addEventListener) {
	window.addEventListener("load",initSearch,false); }
else if (window.attachEvent) {
	window.attachEvent("onload", initSearch); }

function initSearch() {
	// check for quick search table and searchfield
	document.getElementById('id1').style.backgroundColor='#D3D3D3';
	if (!document.getElementById('qs_table')||!document.getElementById('quicksearch')) { return; }

	// load all the rows and sort into arrays
	loadTableData();
	
	//find the query field
	qsfield = document.getElementById('qs_field');

	// previous search term; used for speed optimisation
	prevSearch = '';

	//find statistics location
	stats = document.getElementById('stat');
	setStatistics(-1);
	
	// set up preferences
	initPreferences();

	// shows the searchfield
	document.getElementById('quicksearch').style.display = 'block';
	document.getElementById('qs_field').onkeyup = quickSearch;
}

function loadTableData() {
	// find table and appropriate rows
	searchTable = document.getElementById('qs_table');
	var allRows = searchTable.getElementsByTagName('tbody')[0].getElementsByTagName('tr');

	// split all rows into entryRows and infoRows (e.g. abstract, review, bibtex)
	entryRows = new Array(); infoRows = new Array(); absRows = new Array(); revRows = new Array();

	// get data from each row
	entryRowsData = new Array(); absRowsData = new Array(); revRowsData = new Array(); 
	
	BibTeXKeys = new Array();
	
	for (var i=0, k=0, j=0; i<allRows.length;i++) {
		if (allRows[i].className.match(/entry/)) {
			entryRows[j] = allRows[i];
			entryRowsData[j] = stripDiacritics(getTextContent(allRows[i]));
			allRows[i].id ? BibTeXKeys[j] = allRows[i].id : allRows[i].id = 'autokey_'+j;
			j ++;
		} else {
			infoRows[k++] = allRows[i];
			// check for abstract/review
			if (allRows[i].className.match(/abstract/)) {
				absRows.push(allRows[i]);
				absRowsData[j-1] = stripDiacritics(getTextContent(allRows[i]));
			} else if (allRows[i].className.match(/review/)) {
				revRows.push(allRows[i]);
				revRowsData[j-1] = stripDiacritics(getTextContent(allRows[i]));
			}
		}
	}
	//number of entries and rows
	numEntries = entryRows.length;
	numInfo = infoRows.length;
	numAbs = absRows.length;
	numRev = revRows.length;
}

function quickSearch(){
	
	tInput = qsfield;

	if (tInput.value.length == 0) {
		showAll();
		setStatistics(-1);
		qsfield.className = '';
		return;
	} else {
		t = stripDiacritics(tInput.value);

		if(!searchRegExp) { t = escapeRegExp(t); }
			
		// only search for valid RegExp
		try {
			textRegExp = new RegExp(t,"i");
			closeAllInfo();
			qsfield.className = '';
		}
			catch(err) {
			prevSearch = tInput.value;
			qsfield.className = 'invalidsearch';
			return;
		}
	}
	
	// count number of hits
	var hits = 0;

	// start looping through all entry rows
	for (var i = 0; cRow = entryRows[i]; i++){

		// only show search the cells if it isn't already hidden OR if the search term is getting shorter, then search all
		if(cRow.className.indexOf('noshow')==-1 || tInput.value.length <= prevSearch.length){
			var found = false; 

			if (entryRowsData[i].search(textRegExp) != -1 || BibTeXKeys[i].search(textRegExp) != -1){ 
				found = true;
			} else {
				if(searchAbstract && absRowsData[i]!=undefined) {
					if (absRowsData[i].search(textRegExp) != -1){ found=true; } 
				}
				if(searchReview && revRowsData[i]!=undefined) {
					if (revRowsData[i].search(textRegExp) != -1){ found=true; } 
				}
			}
			
			if (found){
				cRow.className = 'entry show';
				hits++;
			} else {
				cRow.className = 'entry noshow';
			}
		}
	}

	// update statistics
	setStatistics(hits)
	
	// set previous search value
	prevSearch = tInput.value;
}


// Strip Diacritics from text
// http://stackoverflow.com/questions/990904/javascript-remove-accents-in-strings

// String containing replacement characters for stripping accents 
var stripstring = 
    'AAAAAAACEEEEIIII'+
    'DNOOOOO.OUUUUY..'+
    'aaaaaaaceeeeiiii'+
    'dnooooo.ouuuuy.y'+
    'AaAaAaCcCcCcCcDd'+
    'DdEeEeEeEeEeGgGg'+
    'GgGgHhHhIiIiIiIi'+
    'IiIiJjKkkLlLlLlL'+
    'lJlNnNnNnnNnOoOo'+
    'OoOoRrRrRrSsSsSs'+
    'SsTtTtTtUuUuUuUu'+
    'UuUuWwYyYZzZzZz.';

function stripDiacritics(str){

    if(noSquiggles==false){
        return str;
    }

    var answer='';
    for(var i=0;i<str.length;i++){
        var ch=str[i];
        var chindex=ch.charCodeAt(0)-192;   // Index of character code in the strip string
        if(chindex>=0 && chindex<stripstring.length){
            // Character is within our table, so we can strip the accent...
            var outch=stripstring.charAt(chindex);
            // ...unless it was shown as a '.'
            if(outch!='.')ch=outch;
        }
        answer+=ch;
    }
    return answer;
}

// http://stackoverflow.com/questions/3446170/escape-string-for-use-in-javascript-regex
// NOTE: must escape every \ in the export code because of the JabRef Export...
function escapeRegExp(str) {
  return str.replace(/[-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, "\\$&");
}

function toggleInfo(articleid,info) {

	var entry = document.getElementById(articleid);
	var abs = document.getElementById('abs_'+articleid);
	var rev = document.getElementById('rev_'+articleid);
	var bib = document.getElementById('bib_'+articleid);
	
	if (abs && info == 'abstract') {
		abs.className.indexOf('noshow') == -1?abs.className = 'abstract noshow':abs.className = 'abstract show';
	} else if (rev && info == 'review') {
		rev.className.indexOf('noshow') == -1?rev.className = 'review noshow':rev.className = 'review show';
	} else if (bib && info == 'bibtex') {
		bib.className.indexOf('noshow') == -1?bib.className = 'bibtex noshow':bib.className = 'bibtex show';
	} else { 
		return;
	}

	// check if one or the other is available
	var revshow; var absshow; var bibshow;
	(abs && abs.className.indexOf('noshow') == -1)? absshow = true: absshow = false;
	(rev && rev.className.indexOf('noshow') == -1)? revshow = true: revshow = false;	
	(bib && bib.className.indexOf('noshow') == -1)? bibshow = true: bibshow = false;
	
	// highlight original entry
	if(entry) {
		if (revshow || absshow || bibshow) {
		entry.className = 'entry highlight show';
		} else {
		entry.className = 'entry show';
		}
	}
	
	// When there's a combination of abstract/review/bibtex showing, need to add class for correct styling
	if(absshow) {
		(revshow||bibshow)?abs.className = 'abstract nextshow':abs.className = 'abstract';
	} 
	if (revshow) {
		bibshow?rev.className = 'review nextshow': rev.className = 'review';
	}	
	
}

function setStatistics (hits) {
	if(hits < 0) { hits=numEntries; }
	if(stats) { stats.firstChild.data = hits + '/' + numEntries}
}

function getTextContent(node) {
	// Function written by Arve Bersvendsen
	// http://www.virtuelvis.com
	
	if (node.nodeType == 3) {
	return node.nodeValue;
	} // text node
	if (node.nodeType == 1 && node.className != "infolinks") { // element node
	var text = [];
	for (var chld = node.firstChild;chld;chld=chld.nextSibling) {
		text.push(getTextContent(chld));
	}
	return text.join("");
	} return ""; // some other node, won't contain text nodes.
}

function showAll(){
	closeAllInfo();
	for (var i = 0; i < numEntries; i++){ entryRows[i].className = 'entry show'; }
}

function closeAllInfo(){
	for (var i=0; i < numInfo; i++){
		if (infoRows[i].className.indexOf('noshow') ==-1) {
			infoRows[i].className = infoRows[i].className + ' noshow';
		}
	}
}

function clearQS() {
	qsfield.value = '';
	showAll();
}

function redoQS(){
	showAll();
	quickSearch(qsfield);
}

function updateSetting(obj){
	var option = obj.id;
	var checked = obj.value;

	switch(option)
	 {
	 case "opt_searchAbs":
	   searchAbstract=!searchAbstract;
	   redoQS();
	   break;
	 case "opt_searchRev":
	   searchReview=!searchReview;
	   redoQS();
	   break;
	 case "opt_useRegExp":
	   searchRegExp=!searchRegExp;
	   redoQS();
	   break;
	 case "opt_noAccents":
	   noSquiggles=!noSquiggles;
	   loadTableData();
	   redoQS();
	   break;
	 }
}

function initPreferences(){
	if(searchAbstract){document.getElementById("opt_searchAbs").checked = true;}
	if(searchReview){document.getElementById("opt_searchRev").checked = true;}
	if(noSquiggles){document.getElementById("opt_noAccents").checked = true;}
	if(searchRegExp){document.getElementById("opt_useRegExp").checked = true;}
	
	if(numAbs==0) {document.getElementById("opt_searchAbs").parentNode.style.display = 'none';}
	if(numRev==0) {document.getElementById("opt_searchRev").parentNode.style.display = 'none';}
}

function toggleSettings(){
	var togglebutton = document.getElementById('showsettings');
	var settings = document.getElementById('settings');
	
	if(settings.className == "hidden"){
		settings.className = "show";
		togglebutton.innerText = "close settings";
		togglebutton.textContent = "close settings";
	}else{
		settings.className = "hidden";
		togglebutton.innerText = "settings...";		
		togglebutton.textContent = "settings...";
	}
}

function toggleUsage(){
	var togglebutton = document.getElementById('showusage');
	var usage = document.getElementById('usage');
	
	if(usage.className == "hiddenusage"){
		usage.className = "show";
		togglebutton.innerText = "close usage";
		togglebutton.textContent = "close usage";
		usage.innerText = "Usage: Specific topics are colored in grey. Click on these to exclude collapsed ones from search. Press enter on search bar or reload to reset to the original configuration.";
		usage.textContent = "Usage: Specific topics are colored in grey. Click on these to exclude collapsed ones from search. Press enter on search bar or reload to reset to the original configuration.";
	}else{
		usage.className = "hiddenusage";
		togglebutton.innerText = "usage...";		
		togglebutton.textContent = "usage...";
		usage.innerText = "";		
		usage.textContent = "";
	}
}

$(document).ready(function() {

    function getChildren($row) {
        var children = [];
        while($row.next().hasClass('entry')) {
            children.push($row.next());
            $row = $row.next();
            $row = $row.next();
            $row = $row.next();
            $row = $row.next();
        }          
        return children;
    }        

    $('.parent').on('click', function() {
    
        var children = getChildren($(this));
        $.each(children, function() {
            $(this).toggle();
        })
    });
    
})

</script>
<style type="text/css">

body { background-color: white; font-family: Arial, sans-serif; font-size: 13px; line-height: 1.2; padding: 1em; color: #2E2E2E; margin: auto 2em; }

form#quicksearch { width: auto; border-style: solid; border-color: gray; border-width: 1px 0px; padding: 0.7em 0.5em; display:none; position:relative; }
span#searchstat {padding-left: 1em;}

div#settings { margin-top:0.7em; /* border-bottom: 1px transparent solid; background-color: #efefef; border: 1px grey solid; */ }
div#settings ul {margin: 0; padding: 0; }
div#settings li {margin: 0; padding: 0 1em 0 0; display: inline; list-style: none; }
div#settings li + li { border-left: 2px #efefef solid; padding-left: 0.5em;}
div#settings input { margin-bottom: 0px;}

div#settings.hidden {display:none;}

#showsettings { border: 1px grey solid; padding: 0 0.5em; float:right; line-height: 1.6em; text-align: right; }
#showsettings:hover { cursor: pointer; }

#showusage { border: 1px grey solid; padding: 0 0.5em; float:right; line-height: 1.6em; text-align: right; }
#showusage:hover { cursor: pointer; }

.invalidsearch { background-color: red; }
input[type="button"] { background-color: #efefef; border: 1px #2E2E2E solid;}

table { width: 100%; empty-cells: show; border-spacing: 0em 0.2em; margin: 1em 0em; border-style: none; }
th, td { border: 1px gray solid; border-width: 1px 1px; padding: 0.5em; vertical-align: top; text-align: left; }
th { background-color: #efefef; }
td + td, th + th { border-left: none; }

td a { color: navy; text-decoration: none; }
td a:hover  { text-decoration: underline; }

tr.noshow { display: none;}
tr.highlight td { background-color: #EFEFEF; border-top: 2px #2E2E2E solid; font-weight: bold; }
tr.abstract td, tr.review td, tr.bibtex td { background-color: #EFEFEF; text-align: justify; border-bottom: 2px #2E2E2E solid; }
tr.nextshow td { border-bottom: 1px gray solid; }

tr.bibtex pre { width: 100%; overflow: auto; white-space: pre-wrap;}
p.infolinks { margin: 0.3em 0em 0em 0em; padding: 0px; }

@media print {
	p.infolinks, #qs_settings, #quicksearch, t.bibtex { display: none !important; }
	tr { page-break-inside: avoid; }
}
</style>
</head>
<body>

<form action="" id="quicksearch">
<input type="text" id="qs_field" autocomplete="off" placeholder="Type to search..." /> <input type="button" onclick="clearQS()" value="clear" />
<span id="searchstat">Matching entries: <span id="stat">0</span></span>
<div id="showsettings" onclick="toggleSettings()">settings...</div>
<div id="showusage" onclick="toggleUsage()">usage...</div>
<br>
<br>
<div id="usage" class="hiddenusage">
</div>
<div id="settings" class="hidden">
<ul>
<li><input type="checkbox" class="search_setting" id="opt_searchAbs" onchange="updateSetting(this)"><label for="opt_searchAbs"> include abstract</label></li>
<li><input type="checkbox" class="search_setting" id="opt_searchRev" onchange="updateSetting(this)"><label for="opt_searchRev"> include review</label></li>
<li><input type="checkbox" class="search_setting" id="opt_useRegExp" onchange="updateSetting(this)"><label for="opt_useRegExp"> use RegExp</label></li>
<li><input type="checkbox" class="search_setting" id="opt_noAccents" onchange="updateSetting(this)"><label for="opt_noAccents"> ignore accents</label></li>
</ul>
</div>
</form>
<table id="qs_table" border="1">
<thead><tr><th width="3%">Index</th><th width="20%">Author</th><th width="30%">Title</th><th width="5%">Year</th><th width="27%">Journal/Proceedings</th><th width="10%">Reftype</th><th width="5%">DOI/URL</th></tr></thead>
<tbody>

<tr id="id1" class="parent">
     <td colspan="7">Topics</td>
</tr>

<tr id="id2" class="parent">
	<td>1 </td>
    <td colspan="6"> Database: </td>
</tr>

<tr id="stereo_database1" class="entry">
	<td>1. a</td>
	<td colspan="6"><a href="http://vision.middlebury.edu/stereo/data/scenes2014/"> Stereo Database 1</a> &nbsp;</td>
</tr>

<tr id="stereo_database2" class="entry">
	<td>1. b</td>
	<td colspan="6"><a href="https://github.com/LouisFoucard/DepthMap_dataset"> Stereo Database 2 created using blender</a> &nbsp;</td>
</tr>

<tr id="Scharstein" class="entry">
	<td>1. c</td>
	<td>Scharstein</td>
	<td>High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth <p class="infolinks">[<a href="javascript:toggleInfo('Scharstein','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Scharstein','review')">Review</a>] [<a href="javascript:toggleInfo('Scharstein','bibtex')">BibTeX</a>]</p></td>
	<td>2014</td>
	<td>GCPR</td>
	<td>INPROCEEDINGS</td>
	<td><a href="http://www.cs.middlebury.edu/~schar/papers/datasets-gcpr2014.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Scharstein" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
We present a structured lighting system for creating highresolution
stereo datasets of static indoor scenes with highly accurate
ground-truth disparities. The system includes novel techniques for effi-
cient 2D subpixel correspondence search and self-calibration of cameras
and projectors with modeling of lens distortion. Combining disparity
estimates from multiple projector positions we are able to achieve a disparity
accuracy of 0.2 pixels on most observed surfaces, including in halfoccluded
regions. We contribute 33 new 6-megapixel datasets obtained
with our system and demonstrate that they present new challenges for
the next generation of stereo algorithms
	</ul>
	</td>
</tr>
<tr id="rev_Scharstein" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>With bundle adjustment perfect images are produced. </li>
	  <li>Block diagram of how images are produced. (various techniques shown)</li>
	</ul></td>
</tr>
<tr id="bib_Scharstein" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@inproceedings{conf/dagm/ScharsteinHKKNWW14,
  added-at = {2014-10-15T00:00:00.000+0200},
  author = {Scharstein, Daniel and Hirschmüller, Heiko and Kitajima, York and Krathwohl, Greg and Nesic, Nera and Wang, Xi and Westling, Porter},
  biburl = {http://www.bibsonomy.org/bibtex/2c6647ae773975aa5e48a8b29737478ab/dblp},
  booktitle = {GCPR},
  crossref = {conf/dagm/2014},
  editor = {Jiang, Xiaoyi and Hornegger, Joachim and Koch, Reinhard},
  ee = {http://dx.doi.org/10.1007/978-3-319-11752-2_3},
  interhash = {5474bf4e60bdf8fe8b595af37dfd7367},
  intrahash = {c6647ae773975aa5e48a8b29737478ab},
  isbn = {978-3-319-11751-5},
  keywords = {dblp},
  pages = {31-42},
  publisher = {Springer},
  series = {Lecture Notes in Computer Science},
  timestamp = {2015-06-18T22:44:59.000+0200},
  title = {High-Resolution Stereo Datasets with Subpixel-Accurate Ground Truth.},
  url = {http://dblp.uni-trier.de/db/conf/dagm/gcpr2014.html#ScharsteinHKKNWW14},
  volume = 8753,
  year = 2014
}
</pre>
</td>
</tr>

<tr id="id3" class="parent">
	<td>2 </td>
    <td colspan="6"> Deep Learning: </td>
</tr>

<tr id="Zeiler" class="entry">
	<td>2. a</td>
	<td>Eigen, David and Puhrsch, Christian and Fergus, Rob</td>
	<td>Depth Map Prediction from a Single Image using a Multi-Scale Deep Network <p class="infolinks">[<a href="javascript:toggleInfo('Zeiler','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Zeiler','review')">Review</a>] [<a href="javascript:toggleInfo('Zeiler','bibtex')">BibTeX</a>]</p></td>
	<td>2014</td>
	<td>book</td>
	<td>incollection</td>
	<td><a href="https://arxiv.org/pdf/1406.2283v1.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Zeiler" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
Predicting depth is an essential component in understanding the 3D geometry of
a scene. While for stereo images local correspondence suffices for estimation,
finding depth relations from a single image is less straightforward, requiring integration
of both global and local information from various cues. Moreover, the
task is inherently ambiguous, with a large source of uncertainty coming from the
overall scale. In this paper, we present a new method that addresses this task by
employing two deep network stacks: one that makes a coarse global prediction
based on the entire image, and another that refines this prediction locally. We also
apply a scale-invariant error to help measure depth relations rather than scale. By
leveraging the raw datasets as large sources of training data, our method achieves
state-of-the-art results on both NYU Depth and KITTI, and matches detailed depth
boundaries without the need for superpixelation.
	</ul>
	</td>
</tr>
<tr id="rev_Zeiler" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>Uses CNNs to extract local and global component in a very efficient and simple way. </li>
	</ul></td>
</tr>
<tr id="bib_Zeiler" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@incollection{NIPS2014_5539,
title = {Depth Map Prediction from a Single Image using a Multi-Scale Deep Network},
author = {Eigen, David and Puhrsch, Christian and Fergus, Rob},
booktitle = {Advances in Neural Information Processing Systems 27},
editor = {Z. Ghahramani and M. Welling and C. Cortes and N. D. Lawrence and K. Q. Weinberger},
pages = {2366--2374},
year = {2014},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf}
}
</pre>
</td>
</tr>

<tr id="Fischer" class="entry">
	<td>2. b</td>
	<td>A. Dosovitskiy and P. Fischer</td>
	<td>FlowNet: Learning Optical Flow with Convolutional Networks <p class="infolinks">[<a href="javascript:toggleInfo('Fischer','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Fischer','review')">Review</a>] [<a href="javascript:toggleInfo('Fischer','bibtex')">BibTeX</a>]</p></td>
	<td>2015</td>
	<td>ICCV</td>
	<td>InProceedings</td>
	<td><a href="https://arxiv.org/pdf/1504.06852v2.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Fischer" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
Convolutional neural networks (CNNs) have recently
been very successful in a variety of computer vision tasks,
especially on those linked to recognition. Optical flow estimation
has not been among the tasks where CNNs were successful.
In this paper we construct appropriate CNNs which
are capable of solving the optical flow estimation problem
as a supervised learning task. We propose and compare
two architectures: a generic architecture and another one
including a layer that correlates feature vectors at different
image locations.
Since existing ground truth datasets are not sufficiently
large to train a CNN, we generate a synthetic Flying Chairs
dataset. We show that networks trained on this unrealistic
data still generalize very well to existing datasets such as
Sintel and KITTI, achieving competitive accuracy at frame
rates of 5 to 10 fps.

	</ul>
	</td>
</tr>
<tr id="rev_Fischer" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>Uses CNN for optical flow. </li>
	</ul></td>
</tr>
<tr id="bib_Fischer" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@InProceedings{DFIB15,
  author       = "A. Dosovitskiy and P. Fischer and E. Ilg and P. H{\"a}usser and C. Haz\ırba\ş and V. Golkov and P. v.d. Smagt and D. Cremers and T. Brox",
  title        = "FlowNet: Learning Optical Flow with Convolutional Networks",
  booktitle    = "IEEE International Conference on Computer Vision (ICCV)",
  month        = "Dec",
  year         = "2015",
  url          = "http://lmb.informatik.uni-freiburg.de//Publications/2015/DFIB15"
}
</pre>
</td>
</tr>

<tr id="Shen" class="entry">
	<td>2. c</td>
	<td>B. Li and  C. Shen and  Y. Dai</td>
	<td>Depth and surface normal estimation from monocular images using regression on deep features and hierarchical {CRFs}<p class="infolinks">[<a href="javascript:toggleInfo('Shen','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Shen','review')">Review</a>] [<a href="javascript:toggleInfo('Shen','bibtex')">BibTeX</a>]</p></td>
	<td>2015</td>
	<td>CVPR</td>
	<td>InProceedings</td>
	<td><a href="https://arxiv.org/pdf/1504.06852v2.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Shen" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
Convolutional neural networks (CNNs) have recently
been very successful in a variety of computer vision tasks,
especially on those linked to recognition. Optical flow estimation
has not been among the tasks where CNNs were successful.
In this paper we construct appropriate CNNs which
are capable of solving the optical flow estimation problem
as a supervised learning task. We propose and compare
two architectures: a generic architecture and another one
including a layer that correlates feature vectors at different
image locations.
Since existing ground truth datasets are not sufficiently
large to train a CNN, we generate a synthetic Flying Chairs
dataset. We show that networks trained on this unrealistic
data still generalize very well to existing datasets such as
Sintel and KITTI, achieving competitive accuracy at frame
rates of 5 to 10 fps.

	</ul>
	</td>
</tr>
<tr id="rev_Shen" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>Uses CRFs for depth estimation. </li>
	</ul></td>
</tr>
<tr id="bib_Shen" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
 @inproceedings{CVPR15h,
   author    = "B. Li and  C. Shen and  Y. Dai and  A. {van den Hengel} and  M. He",
   title     = "Depth and surface normal estimation from monocular images using regression on deep features and hierarchical {CRFs}",
   booktitle = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR'15)",
   year      = "2015",
 }
</pre>
</td>
</tr>

<tr id="LeCun" class="entry">
	<td>2. d</td>
	<td>Jure Zbontar and Yann LeCun</td>
	<td>Stereo Matching by Training a Convolutional Neural Network to Compare
       <br>Image Patches<p class="infolinks">[<a href="javascript:toggleInfo('LeCun','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('LeCun','review')">Review</a>] [<a href="javascript:toggleInfo('LeCun','bibtex')">BibTeX</a>]</p></td>
	<td>2016</td>
	<td>CoRR</td>
	<td>article</td>
	<td><a href="https://arxiv.org/pdf/1510.05970v2.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_LeCun" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
We present a method for extracting depth information from a rectified image pair. Our
approach focuses on the first stage of many stereo algorithms: the matching cost computation.
We approach the problem by learning a similarity measure on small image patches
using a convolutional neural network. Training is carried out in a supervised manner by
constructing a binary classification data set with examples of similar and dissimilar pairs
of patches. We examine two network architectures for this task: one tuned for speed, the
other for accuracy. The output of the convolutional neural network is used to initialize the
stereo matching cost. A series of post-processing steps follow: cross-based cost aggregation,
semiglobal matching, a left-right consistency check, subpixel enhancement, a median
filter, and a bilateral filter. We evaluate our method on the KITTI 2012, KITTI 2015, and
Middlebury stereo data sets and show that it outperforms other approaches on all three
data sets.
	</ul>
	</td>
</tr>
<tr id="rev_LeCun" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>Uses image patches. </li>
	</ul></td>
</tr>
<tr id="bib_LeCun" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@article{DBLP:journals/corr/ZbontarL15,
  author    = {Jure Zbontar and
               Yann LeCun},
  title     = {Stereo Matching by Training a Convolutional Neural Network to Compare
               Image Patches},
  journal   = {CoRR},
  volume    = {abs/1510.05970},
  year      = {2015},
  url       = {http://arxiv.org/abs/1510.05970},
  timestamp = {Sun, 01 Nov 2015 17:30:45 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/ZbontarL15},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
</pre>
</td>
</tr>

<tr id="id4" class="parent">
	<td>3 </td>
    <td colspan="6"> Related papers: </td>
</tr>

<tr id="Zeiler" class="entry">
	<td>3. a</td>
	<td>Matthew D. Zeiler and Rob Fergus</td>
	<td>Visualizing and Understanding Convolutional Networks <p class="infolinks">[<a href="javascript:toggleInfo('Zeiler','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Zeiler','review')">Review</a>] [<a href="javascript:toggleInfo('Zeiler','bibtex')">BibTeX</a>]</p></td>
	<td>2014</td>
	<td>CoRR</td>
	<td>article</td>
	<td><a href="http://www.matthewzeiler.com/pubs/arxive2013/arxive2013.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Zeiler" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
Large Convolutional Network models have
recently demonstrated impressive classification
performance on the ImageNet benchmark
(Krizhevsky et al., 2012). However
there is no clear understanding of why they
perform so well, or how they might be improved.
In this paper we address both issues.
We introduce a novel visualization technique
that gives insight into the function of intermediate
feature layers and the operation of
the classifier. Used in a diagnostic role, these
visualizations allow us to find model architectures
that outperform Krizhevsky et al. on
the ImageNet classification benchmark. We
also perform an ablation study to discover
the performance contribution from different
model layers. We show our ImageNet model
generalizes well to other datasets: when the
softmax classifier is retrained, it convincingly
beats the current state-of-the-art results on
Caltech-101 and Caltech-256 datasets.

	</ul>
	</td>
</tr>
<tr id="rev_Zeiler" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>With bundle adjustment perfect images are produced. </li>
	  <li>Block diagram of how images are produced. (various techniques shown)</li>
	</ul></td>
</tr>
<tr id="bib_Zeiler" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@article{DBLP:journals/corr/ZeilerF13,
  author    = {Matthew D. Zeiler and
               Rob Fergus},
  title     = {Visualizing and Understanding Convolutional Networks},
  journal   = {CoRR},
  volume    = {abs/1311.2901},
  year      = {2013},
  url       = {http://arxiv.org/abs/1311.2901},
  timestamp = {Tue, 03 Dec 2013 15:04:22 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/ZeilerF13},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
</pre>
</td>
</tr>

<tr id="Andrew" class="entry">
	<td>3. b</td>
	<td>Ashutosh Saxena, Sung H. Chung, and Andrew Y. Ng</td>
	<td>Learning Depth from Single Monocular Images <p class="infolinks">[<a href="javascript:toggleInfo('Andrew','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Andrew','review')">Review</a>] [<a href="javascript:toggleInfo('Andrew','bibtex')">BibTeX</a>]</p></td>
	<td>2005</td>
	<td>NIPS</td>
	<td>INPROCEEDINGS</td>
	<td><a href="https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/saxena-nips-05.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Andrew" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
We consider the task of depth estimation from a single monocular image.
We take a supervised learning approach to this problem, in which
we begin by collecting a training set of monocular images (of unstructured
outdoor environments which include forests, trees, buildings, etc.)
and their corresponding ground-truth depthmaps. Then, we apply supervised
learning to predict the depthmap as a function of the image.
Depth estimation is a challenging problem, since local features alone are
insufficient to estimate depth at a point, and one needs to consider the
global context of the image. Our model uses a discriminatively-trained
Markov Random Field (MRF) that incorporates multiscale local- and
global-image features, and models both depths at individual points as
well as the relation between depths at different points. We show that,
even on unstructured scenes, our algorithm is frequently able to recover
fairly accurate depthmaps.

	</ul>
	</td>
</tr>
<tr id="rev_Andrew" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>Shows first level convolutional kernels. </li>
	</ul></td>
</tr>
<tr id="bib_Andrew" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@INPROCEEDINGS{Saxena05learningdepth,
    author = {Ashutosh Saxena and Sung H. Chung and Andrew Y. Ng},
    title = {Learning depth from single monocular images},
    booktitle = {In NIPS 18},
    year = {2005},
    publisher = {MIT Press}
}
</pre>
</td>
</tr>

<tr id="Ioffe" class="entry">
	<td>3. c</td>
	<td> Sergey Ioffe and Christian Szegedy </td>
	<td>Batch Normalization: Accelerating Deep Network Training by
	<br>Reducing Internal Covariate Shift <p class="infolinks">[<a href="javascript:toggleInfo('Ioffe','abstract')">Abstract</a>] [<a href="javascript:toggleInfo('Ioffe','review')">Review</a>] [<a href="javascript:toggleInfo('Ioffe','bibtex')">BibTeX</a>]</p></td>
	<td>2015</td>
	<td>COrR</td>
	<td>article</td>
	<td><a href="https://arxiv.org/pdf/1502.03167v3.pdf">DOI</a> &nbsp;</td>
</tr>
<tr id="abs_Ioffe" class="abstract noshow">
	<td colspan="7"><b>Abstract</b>: 
	<ul>
Training Deep Neural Networks is complicated by the fact
that the distribution of each layer’s inputs changes during
training, as the parameters of the previous layers change.
This slows down the training by requiring lower learning
rates and careful parameter initialization, and makes it no
toriously hard to train models with saturating nonlinearities.
We refer to this phenomenon as internal covariate
shift, and address the problem by normalizing layer inputs.
Our method draws its strength from making normalization
a part of the model architecture and performing the
normalization for each training mini-batch. Batch Normalization
allows us to use much higher learning rates and
be less careful about initialization. It also acts as a regularizer,
in some cases eliminating the need for Dropout.
Applied to a state-of-the-art image classification model,
Batch Normalization achieves the same accuracy with 14
times fewer training steps, and beats the original model
by a significant margin. Using an ensemble of batchnormalized
networks, we improve upon the best published
result on ImageNet classification: reaching 4.9% top-5
validation error (and 4.8% test error), exceeding the accuracy
of human raters.

	</ul>
	</td>
</tr>
<tr id="rev_Ioffe" class="review noshow">
	<td colspan="7"><b>Review</b>: 
	<ul>
	  <li>Batch normalization as used by the code. </li>
	</ul></td>
</tr>
<tr id="bib_Ioffe" class="bibtex noshow">
<td colspan="7"><b>BibTeX</b>:
<pre>
@article{DBLP:journals/corr/IoffeS15,
  author    = {Sergey Ioffe and
               Christian Szegedy},
  title     = {Batch Normalization: Accelerating Deep Network Training by Reducing
               Internal Covariate Shift},
  journal   = {CoRR},
  volume    = {abs/1502.03167},
  year      = {2015},
  url       = {http://arxiv.org/abs/1502.03167},
  timestamp = {Mon, 02 Mar 2015 14:17:34 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/IoffeS15},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
</pre>
</td>
</tr>


<tr id="id4" class="parent">
	<td>4 </td>
    <td colspan="6"> Links: </td>
</tr>

<tr id="deep-learning-code" class="parent">
	<td>4. a</td>
    <td colspan="6"><a href="https://github.com/LouisFoucard/StereoConvNet"> CNN code for depth estimation</a> &nbsp;</td>
</tr>

</tbody>
</table>