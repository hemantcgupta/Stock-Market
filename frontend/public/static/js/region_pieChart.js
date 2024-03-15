function drawPieChart(Data, DivID) {
    var text = "";

    var width = 200;
    var height = 214;
    var thickness = 40;
    var duration = 750;
    var padding = 10;
    var opacity = .8;
    var opacityHover = 1;
    var otherOpacityOnHover = .8;
    var tooltipMargin = 13;

    var radius = Math.min(width-padding, height-padding) / 2;
    // var color = d3.scale.category10()

     var colors = ['#69c242', '#64bbe3', '#ffcc00', '#ff7300', '#cf2030'];

    // Do not include a domain
    var color = d3.scale.ordinal()
      .range(colors);
    // var color = d3.scale.category10();


    var svg = d3.select("#" + DivID)
    .append('svg')
    .attr('class', 'pie')
    .attr('width', width)
    .attr('height', height);

    var g = svg.append('g')
    .attr('transform', 'translate(' + (width/2) + ',' + (height/2) + ')');

    var arc = d3.svg.arc()  
    .innerRadius(0)
    .outerRadius(radius);

    var pie = d3.layout.pie()
    .value(function(d) { return d.Revenue; })
    .sort(null);

    var path = g.selectAll('path')
      .data(pie(Data))
      .enter()
      .append("g")  
      .append('path')
      .attr('d', arc)
      .attr('fill', (d,i) => color(i))
      .style('opacity', opacity)
      .style('stroke', 'white')
      .on("mouseover", function(d) {
          d3.selectAll('path')
            .style("opacity", otherOpacityOnHover);
          d3.select(this) 
            .style("opacity", opacityHover)
            .transition()
      .duration(500)
      .ease('elastic')
      .attr('transform', function (d) {
      var dist = 10;
      d.midAngle = ((d.endAngle - d.startAngle) / 2) + d.startAngle;
      var x = Math.sin(d.midAngle) * dist;
      var y = -Math.cos(d.midAngle) * dist;
      return 'translate(' + x + ',' + y + ')';
  });


    var total = d3.sum(Data.map(function(d) { // calculate the total number of tickets in the dataset 
      return  d.Revenue; 
    }));
    var percent = Math.round(1000 * d.data.Revenue / total) / 10;

          let g = d3.select(".pie")
            .style("cursor", "pointer")
            .append("g")
            .attr("class", "tooltip_pie")
            .style("opacity", 0)
     
          g.append("text")
            .attr("class", "name-text")
            // .text(`${d.data.Region} (${numberFormat(d.data.Revenue)})`)
            .text(`${d.data.Region} (${percent}%)`)
            .attr('text-anchor', 'middle');
        
          let text = g.select("text");
          let bbox = text.node().getBBox();
          let padding = 2;
          g.insert("rect", "text")
            .attr("x", bbox.x - padding)
            .attr("y", bbox.y - padding)
            .attr("width", bbox.width + (padding*2))
            .attr("height", bbox.height + (padding*2))
            .style("fill", "white")
            .style("opacity", 0.75);
        })
      .on("mousemove", function(d) {
            let mousePosition = d3.mouse(this);
            let x = mousePosition[0] + width/2;
            let y = mousePosition[1] + height/2 - tooltipMargin;
        
            let text = d3.select('.tooltip_pie text');
            let bbox = text.node().getBBox();
            if(x - bbox.width/2 < 0) {
              x = bbox.width/2;
            }
            else if(width - x - bbox.width/2 < 0) {
              x = width - bbox.width/2;
            }
        
            if(y - bbox.height/2 < 0) {
              y = bbox.height + tooltipMargin * 2;
            }
            else if(height - y - bbox.height/2 < 0) {
              y = height - bbox.height/2;
            }
        
            d3.select('.tooltip_pie')
              .style("opacity", 1)
              .attr('transform',`translate(${x}, ${y})`);
        })
      .on("mouseout", function(d) {   
          d3.select(".pie")
            .style("cursor", "none")  
            .select(".tooltip_pie").remove();
        d3.selectAll('path')
            .style("opacity", opacity);
        
        d3.select(this) 
            .transition()
      .duration(500)
      .ease('bounce')
      .attr('transform', 'translate(0,0)')


        })
      .on("touchstart", function(d) {
          d3.select(".pie")
            .style("cursor", "none");    
      })
      .each(function(d, i) { this._current = i; });

    let legend = d3.select("#" + DivID).append('div')
    			.attr('class', 'legend')
    			.style('padding-top', '20%');

    let keys = legend.selectAll('.key')
    			.data(Data)
    			.enter().append('div')
    			.attr('class', 'key')
    			.style('display', 'flex')
    			.style('align-items', 'center')
    			.style('margin-right', '20px');

    		keys.append('div')
    			.attr('class', 'symbol')
    			.style('height', '10px')
    			.style('width', '10px')
    			.style('margin', '5px 5px')
    			.style('background-color', (d, i) => color(i));

    		keys.append('div')
    			.attr('class', 'name')
    			.text(d => `${d.Region} (${numberFormat(d.Revenue)})`);

    		// keys.exit().remove();

  }