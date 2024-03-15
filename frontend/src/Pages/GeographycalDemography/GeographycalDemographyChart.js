import React, { Component } from "react";
import TopMenuBar from '../../Component/TopMenuBar';
import FusionCharts from "fusioncharts";
import Charts from "fusioncharts/fusioncharts.charts";
import APIServices from '../.././API/apiservices'
import ReactFC from "react-fusioncharts";
import cookieStorage from '../../Constants/cookie-storage';
import String from '../.././Constants/validator';
import $ from 'jquery';
import eventApi from '../../API/eventApi'
import './GeographicalDemo.scss'

ReactFC.fcRoot(FusionCharts, Charts);

const apiServices = new APIServices();

// const rangeValue = {
//   from: {
//     year: new Date().getFullYear(),
//     month: 1
//   },
//   to: {
//     year: new Date().getFullYear(),
//     month: (new Date().getMonth() + 1)
//   }
// }

// let rangeValue = JSON.parse(window.localStorage.getItem('rangeValue'))
class DemographyDomInt extends Component {
  constructor(props) {
    super(props);
    this.state = {
      regionId: 'Null',
      countryId: 'Null',
      // startDate: this.formatDate(new Date(rangeValue.from.year, rangeValue.from.month - 1, 1)),
      // endDate: this.formatDate(new Date(rangeValue.to.year, rangeValue.to.month, 0)),
      startDate: 'Null',
      endDate: 'Null',
      GeographicalChartConfig: {
        type: "mscolumn2d",
        renderAt: "chart-container",
        width: "100%",
        height: "520",
        dataFormat: "json",
        dataSource: {
          chart: {
            theme: "candy",
            caption: "Website Trafic (Visits)",
            numberPrefix: "",
            plotFillAlpha: "80",
            divLineIsDashed: "1",
            divLineDashLen: "1",
            divLineGapLen: "1",
            // rotateLabels: "45deg",
            labelDisplay: "rotate",
            slantLabel: "1"
          },
          categories: [{
            "category":
              []
          }],
          dataset: [
            {
              seriesname: "Previous Year",
              color: "#0f5298",
              data: [],
            },
            {
              seriesname: "Curent Year",
              color: "#d5f3fe",
              data: [],
            },
          ],
          // "trendlines": [{
          //   "line": [{
          //     "startvalue": "12250",
          //     "color": "#5D62B5",
          //     "displayvalue": "Previous{br}Average",
          //     "valueOnRight": "1",
          //     "thickness": "1",
          //     "showBelow": "1",
          //     "tooltext": "Previous year quarterly target  : $13.5K"
          //   }, {
          //     "startvalue": "25950",
          //     "color": "#29C3BE",
          //     "displayvalue": "Current{br}Average",
          //     "valueOnRight": "1",
          //     "thickness": "1",
          //     "showBelow": "1",
          //     "tooltext": "Current year quarterly target  : $23K"
          //   }]
          // }]
        },
      },
    };
  }

  sendEvent = (token) => {
    var eventData = {
      event_id: "2",
      description: "User viewed Geographical Chart Page",
      where_path: "/geographyInReport",
      page_name: "Geography Chart Page"
    }
    eventApi.sendEventWithHeader(eventData, token)
  }

  componentDidMount() {
    const token = cookieStorage.getCookie('Authorization')
    this.sendEvent(token);
    this.getPOSSelectedGeographicalDetails();
    // this.getGraphData();

  }

  getPOSSelectedGeographicalDetails = ($event) => {
    let { regionId, countryId, startDate, endDate } = this.state
    let rangeValue = JSON.parse(window.localStorage.getItem('rangeValue'))
    let startDateVar = this.formatDate(new Date(rangeValue.from.year, rangeValue.from.month - 1, 1));
    let endDateVar = this.formatDate(new Date(rangeValue.to.year, rangeValue.to.month, 0));
    let RegionSelected = window.localStorage.getItem('RegionSelected')
    let CountrySelected = window.localStorage.getItem('CountrySelected')
    this.setState({
      regionId: RegionSelected === null || RegionSelected === 'Null' || RegionSelected === '' ? 'Null' : JSON.parse(RegionSelected),
      countryId: CountrySelected === null || CountrySelected === 'Null' || CountrySelected === '' ? 'Null' : JSON.parse(CountrySelected),
      startDate: startDateVar,
      endDate: endDateVar,
    }
      , () => this.getGraphData())
  }


  getGraphData = () => {
    var self = this
    const { startDate, endDate, regionId, countryId, dataset, categories } = this.state;
    apiServices.getTopGeographicalDemographyChartData(startDate, endDate, regionId, countryId,).then(function (result) {
      if (result) {
        var categories = result[0].categories
        var dataset = result[0].dataset
        self.setState({ dataset: dataset, categories: categories })
        let GeographicalChartConfig = self.state.GeographicalChartConfig
        GeographicalChartConfig.dataSource.categories[0].category = categories
        GeographicalChartConfig.dataSource.dataset[0].data = dataset[1].data
        GeographicalChartConfig.dataSource.dataset[1].data = dataset[0].data
        self.setState({ GeographicalChartConfig: GeographicalChartConfig })
      }
    });
  }


  formatDate = (d) => {
    let month = '' + (d.getMonth() + 1);
    let day = '' + d.getDate();
    let year = d.getFullYear();

    if (month.length < 2)
      month = '0' + month;
    if (day.length < 2)
      day = '0' + day;

    return [year, month, day].join('-');
  }

  // componentDidMount() {
  //     let props = this.props;
  //     let barChartConfigs = this.state.barChartConfigs;
  //     barChartConfigs.dataSource.data = props.datasetOne
  //     this.setState({ barChartConfigs: barChartConfigs })

  // }

  // componentWillReceiveProps = (props) => {
  //     let barChartConfigs = this.state.barChartConfigs;
  //     barChartConfigs.dataSource.data = props.datasetOne
  //     this.setState({ barChartConfigs: barChartConfigs })
  // }

  render() {
    // console.log(this.state.GeographicalChartConfig, 'config')
    return (
      <div className="x_panel tile">
        <TopMenuBar dashboardPath={'/demographyDashboard'} getPOSSelectedGeographicalDetails={this.getPOSSelectedGeographicalDetails} {...this.props} />
        <div className="x_title reduce-margin">
          <h2 className="responsive-size">Top 20 Geographycal Report </h2>


        </div>
        {/* {isLoading ? */}

        {/* <h5 style={{ textAlign: 'center', margin: '17%' }}>No data to show</h5> : */}
        <div className="chart-container">
          <div id="segmentation"></div>
          <ReactFC {...this.state.GeographicalChartConfig} />
          {this.state.GeographicalChartConfig.dataSource.dataset.length > 0 ? <div className='hide-watermark'></div> : ''}
          {/* <RenderChart GeographicalChartConfig={this.state.GeographicalChartConfig} /> */}
        </div>
      </div>
    );
  }
}

// const RenderChart = (props) => {
//   console.log('render')
//   const { GeographicalChartConfig } = props
//   return <ReactFC {...GeographicalChartConfig} />
// }

export default DemographyDomInt;