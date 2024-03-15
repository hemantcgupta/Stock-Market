import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
//import MutliBarGraphLegends from '../../../../Component/MutliBarGraphLegends';
//import * as d3 from "d3";
//import Chart from '../../../../Component/DrawBarChart';
import Chart from "react-apexcharts";
import Spinners from "../../../../spinneranimation";
import '../DemographyDashboard.scss';

const apiServices = new APIServices();

class DemographyAgeChart extends Component {
  constructor(props) {
    super();
    this.state = {
      ageGroupData: [],
      colors: ["#0571b0", "#92c5de", '#46b6ed', "#4CAF50", "#d5d5d5", "#92c5de", "#0571b0"],
      headerName: 'Unique',
      isLoading: false,
      series: [{
        data: []
      }],
      options: {
        chart: {
          type: 'bar',
          height: 350
        },
        plotOptions: {
          bar: {
            borderRadius: 1,
            horizontal: true,
          }
        },
        dataLabels: {
          enabled: false
        },
        xaxis: {
          categories: [],
        }
      },

    };
  }

  componentWillReceiveProps = (props) => {
    this.getBarchartData(props);
  }

  getBarchartData = (props) => {
    const self = this;
    const { startDate, endDate, regionId, countryId, cityId } = props;
    if (self.state.headerName === 'Unique') {
      this.setState({ isLoading: true, ageGroupData: [] })
      apiServices.getDemographyAgeGroupData(startDate, endDate, regionId, countryId, cityId, 'Unique').then((ageGroupDataResponse) => {
        this.setState({ isLoading: false })
        if (ageGroupDataResponse) {
          const tableData = ageGroupDataResponse[0].Data
          if (tableData.length > 0) {
            this.setState(
              { ageGroupData: this.normalizeData(tableData) }
            );
          }
        }
      });
    } else {
      this.setState({ isLoading: true, ageGroupData: [] })
      apiServices.getDemographyAgeGroupData(startDate, endDate, regionId, countryId, cityId, 'All').then((ageGroupDataResponse) => {
        this.setState({ isLoading: false })
        if (ageGroupDataResponse) {
          const tableData = ageGroupDataResponse[0].Data
          if (tableData.length > 0) {
            this.setState(
              { ageGroupData: this.normalizeData(tableData) }
            );
          }
        }
      });
    }
  }

  normalizeData = (data) => {
    let xValues = data.map(function (d) { return d.label; });
    let yValues = data.map(function (d) { return parseInt(d.value); });
    let normalizedData = data;
    for (var i = 0; i < xValues.length; i++) {
      let categoryObj = { x: xValues[i].toUpperCase(), y: yValues[i] };
      normalizedData.push(categoryObj);
    }
    this.setState(
      {
        series: [
          {
            name: 'Count:',
            data: yValues
          }
        ],
        options: {
          chart: {
            type: 'bar',
            height: 350,
            toolbar: {
              show: false
            }
          },
          stroke: {
            width: 0,
            colors: ['#fff']
          },
          plotOptions: {
            bar: {
              borderRadius: 1,
              horizontal: true,
            }
          },
          dataLabels: {
            enabled: false
          },
          xaxis: {
            categories: xValues,
            labels: {
              style: {
                cssClass: "hidegraphlabels"
              }
            }
          },
          tooltip: {
            style: {
              fontSize: '10px',
              color: "#fff"
            },
            x: {
              show: false,
            },
            y: {
              formatter: undefined,
              title: {
                formatter: (seriesName) => seriesName,
              },
            },
          }
        }
      }
    );
    return normalizedData;
  }

  graphChange(e) {
    this.setState({ headerName: e.target.value }, () => this.getBarchartData(this.props))
  }

  render() {
    const { isLoading, ageGroupData } = this.state;
    return (
      <div className="x_panel tile" style={{ paddingTop: '0px', overflow: 'visible' }}>
        <div className="x_title reduce-margin">
          <h2 className="responsive-size">Age Group</h2>
          <div className="nav navbar-right" style={{ display: "flex", alignItems: "center", marginTop: "-2px" }}
          >
            <select class="select header-dropdown responsive-size" onChange={(e) => this.graphChange(e)}>
              <option>Unique</option>
              <option>All</option>
            </select>

            <ul className="nav navbar-right panel_toolbox">
              <div className='info'><li><i class="fa fa-info" aria-hidden="true"></i></li>
                {/*<MutliBarGraphLegends i={true} data={topfiveODsData} agentsName={agentsName} colors={colors} />*/}
              </div>
              <li onClick={() => this.props.history.push("/demographyAgeGroup")}><i className="fa fa-line-chart"></i></li>
            </ul>
          </div>
        </div>
        {isLoading ? <Spinners /> : ageGroupData.length === 0 ?
          <h5 style={{ textAlign: 'center', margin: '20%' }}>No data to show</h5> :
          <div className='centered-graph'>
            <div id="topODAgent"></div>
            <Chart
              options={this.state.options}
              series={this.state.series}
              type="bar"
              style={{ overflowY: "scroll" }}
            />
            {/*<MutliBarGraphLegends data={topfiveODsData} agentsName={agentsName} colors={colors} />*/}
          </div>}
      </div>
    )
  }
}

export default DemographyAgeChart;