import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import DataTableComponent from '../../../../Component/DataTableComponent'
import TotalRow from '../../../../Component/TotalRow'
import String from '../../../../Constants/validator';

const apiServices = new APIServices();

class RegionWisePerformance extends Component {

  constructor(props) {
    super(props);
    this.state = {
      regionWisePerformance: [],
      totalData: [],
      regionWisePerformanceColumn: [],
      headerName: '',
      loading: false
    };
  }

  componentWillReceiveProps = (props) => {
    var self = this;
    const { startDate, endDate, routeGroup, regionId, countryId, routeId } = props;
    const group = String.removeQuotes(routeGroup)
    this.url = `/route?RouteGroup=${group}`;
    this.getRouteURL(group, regionId, countryId, routeId);
    self.setState({ loading: true, regionWisePerformance: [] })
    apiServices.getRegionWisePerformanceTable(startDate, endDate, routeGroup, regionId, countryId, routeId).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;
        var tableHead = result[0].tableHead;

        self.setState({ regionWisePerformance: rowData, totalData: totalData, regionWisePerformanceColumn: columnName, headerName: tableHead })
      }
    });

  }

  getRouteURL(routeGroup, regionId, countryId, routeId) {
    let leg = window.localStorage.getItem('LegSelected')
    let flight = window.localStorage.getItem('FlightSelected')

    if (regionId !== 'Null') {
      this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}`
    }
    if (countryId !== 'Null') {
      this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}`
    }
    if (routeId !== 'Null') {
      this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}`
    }
    if (leg !== null && leg !== 'Null' && leg !== '') {
      this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}&Leg=${String.removeQuotes(leg)}`
    }
    if (flight !== null && flight !== 'Null' && flight !== '') {
      this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}&Leg=${String.removeQuotes(leg)}&Flight=${String.removeQuotes(flight)}`
    }
  }

  render() {
    const { regionWisePerformance, totalData, regionWisePerformanceColumn, loading, headerName } = this.state;
    return (
      <div>
        <div className="x_title">
          <h2>{`${headerName} PERFORMANCE`}</h2>
          <ul className="nav navbar-right panel_toolbox">
            {/* <li><a className="collapse-link"><i className="fa fa-chevron-up"></i></a>
                </li>
                <li className="dropdown">
                  <a href="#" className="dropdown-toggle" data-toggle="dropdown" role="button" aria-expanded="false"><i className="fa fa-wrench"></i></a>
                  <ul className="dropdown-menu" role="menu">
                    <li><a href="#">Settings 1</a>
                    </li>
                    <li><a href="#">Settings 2</a>
                    </li>
                  </ul>
                </li> */}
            <li onClick={() => this.props.history.push(this.url)}><i className="fa fa-line-chart"></i></li>
            {/* <li><a className="close-link"><i className="fa fa-close"></i></a></li> */}
          </ul>
        </div>
        <div className="x_content">
          <DataTableComponent
            rowData={regionWisePerformance}
            columnDefs={regionWisePerformanceColumn}
            dashboard={true}
            channel={true}
            loading={loading}
            height={'20vh'}
          />
          <TotalRow
            rowData={totalData}
            columnDefs={regionWisePerformanceColumn}
            loading={loading}
            dashboard={true}
            reducingPadding={true}
          />
        </div>
      </div>
    )
  }
}

export default RegionWisePerformance;