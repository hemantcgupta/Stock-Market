function numberFormat(amtData, formatFixed){ // Amount format handle in Multiline chart in all the places
    var num = Math.abs(amtData);
    // if (num >= 1000000000) {
        // formattedNumber = (num / 1000000000).toFixed(formatFixed).replace(/\.0$/, '') + 'G';
    // } else
    if (num >= 1000000) {
        formattedNumber =  (num / 1000000).toFixed(formatFixed).replace(/\.0$/, '') + 'M';
    } else  if (num >= 1000) {
        formattedNumber =  (num / 1000).toFixed(formatFixed).replace(/\.0$/, '') + 'K';
    } else {
        formattedNumber = num;
    } 

    return formattedNumber;
}

function GetMonthName(monthNumber) { // Month Handle request number and return month name...
      var months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
      return months[monthNumber - 1];
}

function indicator_gen(indicatorsData){

    // Indicator data shown on Dashboard Page...
    $.each( indicatorsData, function( key, value ) {
        
        totlaAmount = numberFormat(value.total, 2); // Curreny format in Juqery...
        if(value.id_form == "load_factor"){
            validateAmt = value.total.toFixed(2);
            totlaAmount = validateAmt + " %";
        }
        amountPer = value.per.toFixed(2); // % calculation with two decimal value...
        var strPer = amountPer.slice(0,1); // slice the '-' values for color chage in indicators...
        if(strPer == '-'){ // condition check for color change like indicatores increse or decrese value...
            spanIClass = "red";
        } else {
            spanIClass = "green";            
        }

        htm = '<div class="col-sm-4 col-xs-6 tile_stats_count">';
        htm += '<span class="count_top">' + value.name + '</span>';
        htm += '<div class="count" id=' + value.id_form + '>' + totlaAmount + '</div>';
        htm += '<span class="count_bottom"><i class="'+spanIClass+'" id="revenue_YTD_per">' + amountPer + ' %</i></span>';
        htm += '<div class="progress active" id="indicator_progress" style="width:90%;">';
        htm += '<div class="progress-bar bg-'+spanIClass+'" role="progressbar" aria-valuenow="70" aria-valuemin="0" aria-valuemax="100" style="width:' + amountPer + '%">';
        htm += '</div>';
        htm += '</div>';
        htm += '</div>';

        $(".indicator_data").append(htm);
    });

    // Column split depends on div element in dashboard indicator segment... 
    var divCalcLength = $(".indicator_data > div.tile_stats_count").length;
    var calc = 100/divCalcLength;
    $(".tile_stats_count").css("width", calc+"%");

}