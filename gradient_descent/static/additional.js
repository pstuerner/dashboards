setInterval("MathJax.Hub.Queue(['Typeset',MathJax.Hub])",100);

var waitForPlotly = setInterval( function() {
            if( typeof(window.Plotly) !== "undefined" ){
                MathJax.Hub.Config({ SVG: { font: "STIX-Web" }, displayAlign: "center" });
                MathJax.Hub.Queue(["setRenderer", MathJax.Hub, "SVG"]);
                clearInterval(waitForPlotly);
            }}, 250 );

MathJax.Hub.Config({
    jax: ["input/TeX","output/HTML-CSS"],
    displayAlign: "center"
});
