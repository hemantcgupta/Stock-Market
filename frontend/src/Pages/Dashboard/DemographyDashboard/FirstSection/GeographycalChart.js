import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
//import MutliBarGraphLegends from '../../../../Component/MutliBarGraphLegends';
//import Chart from '../../../../Component/DrawBarChart';
import Chart from "react-apexcharts";
import String from '../../../../Constants/validator';
import Spinners from "../../../../spinneranimation";
import '../DemographyDashboard.scss';


const apiServices = new APIServices();

class GeographycalChart extends Component {
    constructor(props) {
        super();
        this.state = {
            locationVistorData: ['hello'],
            colors: ["#009688", "#4CAF50", "#03f574", "#92c5de", "#0571b0", "#92c5de", "#0571b0"],
            isLoading: false,

            series: [],

            options: {
                xaxis: {
                    labels: {
                        show: false
                    }
                },
                legend: {
                    show: true
                },
                tooltip: {
                    style: {
                        fontSize: '10px',
                    },

                    fixed: {
                        enabled: true,
                        position: 'topLeft',
                        offsetX: 0,
                        offsetY: 0,
                    }
                },
                chart: {
                    type: 'treemap',
                    redrawOnParentResize: true,
                    //width: 200,
                    offsetY: 0,
                    //height: 250,
                    toolbar: {
                        show: false
                    },
                    defaultLocale: 'en'
                },
                colors: [
                    '#3B93A5', '#F7B844', '#ADD8C7', '#EC3C65', '#CDD7B6', '#C1F666', '#D43F97', '#1E5D8C', '#421243', '#7F94B0', '#EF6537', '#C0ADDB'
                ],
                plotOptions: {
                    treemap: {
                        distributed: true,
                        enableShades: false
                    }
                }
            },
        };
    }

    componentWillReceiveProps = (props) => {

        const { startDate, endDate, regionId, countryId, cityId, ODId } = props;
        this.url = '/geographyInReport';
        this.getPOSURL(regionId, countryId, cityId, ODId);
        this.setState({ isLoading: true })
        apiServices.getDemographyTop20LocationVisitors(startDate, endDate, regionId, countryId, cityId).then((locationVistorDataResponse) => {
            this.setState({ isLoading: false })
            if (locationVistorDataResponse) {
                const tableData = locationVistorDataResponse[0].Data;
                if (tableData.length > 20) {
                    tableData.length = 20;
                }
                if (tableData.length > 0) {
                    let xValues = tableData.map(function (d) { return d.label; });
                    let yValues = tableData.map(function (d) { return d.value; });
                    let normalizedData = [];
                    for (var i = 0; i < xValues.length; i++) {
                        let categoryObj = { x: (xValues[i]) + ".", y: parseInt(yValues[i]) };
                        normalizedData.push(categoryObj);
                    }
                    this.setState({ series: [{ name: 'Geography', data: normalizedData }] });
                    // this.setState(
                    //     { locationVistorData: tableData }
                    // );
                }
            }
        });
    }


    normalizeData = (dataset) => {
        let xValues = dataset.map(function (d) { return d.label; });
        let yValues = dataset.map(function (d) { return d.value; });
        let normalizedData = [];
        for (var i = 0; i < xValues.length; i++) {
            let categoryObj = { x: xValues[i], y: yValues[i] };
            normalizedData.push(categoryObj);
        }
        this.setState(
            {
                series: [
                    {
                        data: normalizedData
                    }
                ]
            }
        );
        // console.log('norma data', normalizedData)
        //const cleanedData = normalizedData.slice(0, 20);
        // console.log('cleaned data', cleanedData)
        console.log('states', this.state)
        return dataset;
    }

    getPOSURL(regionId, countryId, cityId, ODId) {
        if (regionId !== 'Null') {
            this.url = `/geographyInReport?Region=${String.removeQuotes(regionId)}`
        }
        if (countryId !== 'Null') {
            this.url = `/geographyInReport?Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}`
        }
        // if (cityId !== 'Null') {
        //     this.url = `/pos?Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&POS=${String.removeQuotes(cityId)}`
        // }
        // if (ODId !== 'Null') {
        //     this.url = `/pos?Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&POS=${String.removeQuotes(cityId)}&${encodeURIComponent('O&D')}=${String.removeQuotes(ODId)}`
        // }
    }

    render() {
        const { isLoading, locationVistorData } = this.state;
        console.log('geog states', this.state)
        return (
            <div className="x_panel tile">
                <div className="x_title reduce-margin">
                    <h2 className='responsive-size'>Top 20 Geographical Location of Visitor</h2>
                    <ul className="nav navbar-right panel_toolbox">
                        <div className='info'><li><i class="fa fa-info" aria-hidden="true"></i></li>

                            {/*<MutliBarGraphLegends i={true} data={performanceData} colors={colors} />*/}
                        </div>
                        <li onClick={() => this.props.history.push(this.url)}><i className="fa fa-line-chart"></i></li>
                    </ul>
                </div>
                {isLoading ?
                    <Spinners /> :
                    locationVistorData.length === 0 ?
                        <h5 style={{ textAlign: 'center', margin: '20%' }}>No data to show</h5> :
                        <div className='centered-graph'>
                            <div id="regionPerformnace2"></div>
                            <Chart
                                options={this.state.options}
                                series={this.state.series}
                                type="treemap"
                                width="95%"
                                style={{ overflowY: "scroll" }}
                            />
                            {/*<MutliBarGraphLegends data={performanceData} colors={colors} />*/}
                        </div>}

            </div>
        )
    }
}

export default GeographycalChart;
