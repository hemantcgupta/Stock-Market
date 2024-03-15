import React, { Component } from "react";
import APIServices from '../../../../API/apiservices';
import DataTableComponent from '../../../../Component/DataTableComponent'
import TotalRow from '../../../../Component/TotalRow'
import String from '../../../../Constants/validator';

const apiServices = new APIServices();

class RegionBudget extends Component {

  constructor(props) {
    super(props);
    this.state = {
      RegionBudget: [],
      totalData: [],
      RegionBudgetColumn: [],
      headerName: '',
      loading: false
    };
  }

  componentWillReceiveProps = (props) => {
    var self = this;
    const { startDate, endDate, routeGroup, regionId, countryId, routeId } = props;
    const group = String.removeQuotes(routeGroup)
    // this.url = `/rps`;
    this.getRouteURL(group, regionId, countryId, routeId);
    self.setState({ loading: true, RegionBudget: [] })
    apiServices.getRegionBudgetTable(startDate, endDate, routeGroup, regionId, countryId, routeId).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;
        var tableHead = result[0].tableHead;

        self.setState({ RegionBudget: rowData, totalData: totalData, RegionBudgetColumn: columnName, headerName: tableHead })
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
    const { RegionBudget, totalData, RegionBudgetColumn, loading, headerName } = this.state;
    return (
      <div>
        <div className="x_title">
          <h2>{`${headerName} Budget`}</h2>
          <ul className="nav navbar-right panel_toolbox">
            <li onClick={() => this.props.history.push(this.url)}><i className="fa fa-line-chart"></i></li>
          </ul>
        </div>
        <div className="x_content">
          <DataTableComponent
            rowData={RegionBudget}
            columnDefs={RegionBudgetColumn}
            dashboard={true}
            channel={true}
            loading={loading}
            height={'20vh'}
          />
          <TotalRow
            rowData={totalData}
            columnDefs={RegionBudgetColumn}
            loading={loading}
            dashboard={true}
            reducingPadding={true}
          />
        </div>
      </div>
    )
  }
}

export default RegionBudget;